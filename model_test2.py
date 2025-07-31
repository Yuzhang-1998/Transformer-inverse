import tensorflow as tf
import argparse
import pickle
from model2 import build_transformer_model
from sklearn.preprocessing import StandardScaler


def split_data(X, y, validation_split=0.2):
    num_samples = tf.shape(X)[0]
    val_count = tf.cast(tf.cast(num_samples, tf.float32) * validation_split, tf.int32)
    idx = tf.random.shuffle(tf.range(num_samples))
    val_idx, train_idx = idx[:val_count], idx[val_count:]
    X_train = tf.gather(X, train_idx)
    y_train = tf.gather(y, train_idx)
    X_val = tf.gather(X, val_idx)
    y_val = tf.gather(y, val_idx)
    return X_train, y_train, X_val, y_val


def train_model(X, y, epochs, batch_size, lr, validation_split, strategy, best_model_path):
    # ????/???
    X_train, y_train, X_val, y_val = split_data(X, y, validation_split)

    # ?? tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
        .shuffle(450) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    # ???
    train_dist = strategy.experimental_distribute_dataset(train_ds)
    val_dist = strategy.experimental_distribute_dataset(val_ds)

    with strategy.scope():
        #model = build_transformer_model()  # ?? output_dim=20
        model=tf.keras.models.load_model('saved_model_2.keras')
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.MeanSquaredError()
        # dummy build
        model(tf.ones((1, X.shape[1], X.shape[2])))

    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = loss_fn(targets, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def distributed_train_step(inputs, targets):
        per_replica_loss = strategy.run(train_step, args=(inputs, targets))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    def val_step(inputs, targets):
        preds = model(inputs, training=False)
        return loss_fn(targets, preds)

    @tf.function
    def distributed_val_step(inputs, targets):
        per_replica_loss = strategy.run(val_step, args=(inputs, targets))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    #best_val_loss = float('inf')
    best_val_loss = tf.constant(0.0001)
    print("【训练开始】共", epochs, "个 epoch。")
    for epoch in range(1, epochs + 1):
        print("\n【Epoch】", epoch, "/", epochs, ": 开始训练...")
        total_train_loss = 0.0
        train_steps = 0
        for x_batch, y_batch in train_dist:
            loss = distributed_train_step(x_batch, y_batch)
            total_train_loss += float(loss)
            train_steps += 1
        avg_train_loss = total_train_loss / train_steps

        # ??
        total_val_loss = 0.0
        val_steps = 0
        for x_batch, y_batch in val_dist:
            loss = distributed_val_step(x_batch, y_batch)
            total_val_loss += float(loss)
            val_steps += 1
        avg_val_loss = total_val_loss / val_steps

        print(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(best_model_path)
            print(f"  Saved improved model to {best_model_path}")

    return model


def main(args):
    strategy = tf.distribute.MirroredStrategy()
    print("GPUs:", strategy.num_replicas_in_sync)

    # ????
    with open('X_beam.pkl', 'rb') as f:
        X = pickle.load(f)  # shape (N, 231, 4)
    with open('Y_beam_2.pkl', 'rb') as f:
        Y = pickle.load(f)  # shape (N, 231, 20)

    # ??? Y
    X=X[:480,:]
    Y=Y[:480,:]
    print(Y.shape)
    N, L, D = Y.shape
    Y_flat = Y.reshape(-1, D)
    scaler = StandardScaler().fit(Y_flat)
    Y = scaler.transform(Y_flat).reshape(N, L, D)

    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    # ??
    train_model(
        X, Y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        validation_split=args.validation_split,
        strategy=strategy,
        best_model_path=args.model_save_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--model_save_path', type=str, default='saved_model_2.keras')
    args = parser.parse_args()
    main(args)

