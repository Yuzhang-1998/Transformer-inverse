import tensorflow as tf
import argparse
import pickle
from model import build_transformer_model  # 假设模型构建代码在 model.py 中
#from model_decoder import build_transformer_model_with_decoder
from sklearn.preprocessing import StandardScaler



# -------------------------------

# 数据拆分函数

# -------------------------------

def split_data(X, y, validation_split=0.2):
    print("【数据拆分】开始拆分数据...")
    num_samples = tf.shape(X)[0]
    val_samples = tf.cast(tf.cast(num_samples, tf.float32) * validation_split, tf.int32)
    indices = tf.random.shuffle(tf.range(num_samples))
    val_indices = indices[:val_samples]
    train_indices = indices[val_samples:]
    X_train = tf.gather(X, train_indices)
    y_train = tf.gather(y, train_indices)
    X_val = tf.gather(X, val_indices)
    y_val = tf.gather(y, val_indices)
    print("【数据拆分】数据拆分完成。")
    return X_train, y_train, X_val, y_val



# -------------------------------

# 模型训练函数（分布式训练）

# -------------------------------

def train_model(X, y, epochs, batch_size, learning_rate, validation_split, strategy,best_model_path):

# def train_model(X, y, epochs, batch_size, learning_rate, validation_split):
    print("【数据处理】开始拆分数据集...")
    X_train, y_train, X_val, y_val = split_data(X, y, validation_split)
    print("【数据处理】构建 tf.data.Dataset 数据集...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=4000, reshuffle_each_iteration=True)\
                                 .batch(batch_size)\
                                 .prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)\
                             .prefetch(tf.data.AUTOTUNE)

    print("【数据处理】将数据集分发到各个 GPU 上...")
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    print("【模型构建】在 strategy.scope 内构建模型、优化器和损失函数...")
    
    with strategy.scope():
        #model = build_transformer_model()
        model=tf.keras.models.load_model('saved_model.keras')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
                # 增加 dummy run，确保所有变量已初始化
        dummy_input = tf.ones((1, 231, 4))
        _ = model(dummy_input)

        print("【模型构建】模型构建完成。")


    # 普通的训练步骤函数

    def train_step_fn(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

        

    # for batch in train_dataset.take(1):
    #     inputs, targets = batch
    #     inputs = tf.cast(inputs, tf.float32)
    #     targets = tf.cast(targets, tf.float32)
    #     loss_val = train_step_fn(inputs, targets)
    #     print("Single batch loss:", loss_val.numpy())



    @tf.function
    def distributed_train_step(inputs, targets):
        per_replica_loss = strategy.run(train_step_fn, args=(inputs, targets))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)



    # 普通的验证步骤函数

    def val_step_fn(inputs, targets):
        predictions = model(inputs, training=False)
        loss = loss_fn(targets, predictions)
        return loss



    @tf.function

    def distributed_val_step(inputs, targets):
        per_replica_loss = strategy.run(val_step_fn, args=(inputs, targets))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    # #debug

    # def run_epochs_debug(epochs):
    #     print("【训练开始】共", epochs, "个 epoch。")
    #     for epoch in range(epochs):
    #         print("\n【Epoch】", epoch+1, "/", epochs, ": 开始训练...")
    #         total_train_loss = tf.zeros( [],dtype=tf.dtypes.float32)
    #         train_steps = tf.zeros( [],dtype=tf.dtypes.int32)
    #         for batch in train_dataset.take(-1):
    #             inputs, targets = batch
    #             # inputs=tf.cast(inputs, tf.float32)
    #             # targets=tf.cast(targets, tf.float32)
    #             # print(inputs,targets)
    #             loss = distributed_train_step(inputs, targets)
    #             print('loss:',loss)
    #             total_train_loss += loss  # 转换为 numpy
    #             train_steps += 1
    #             print("  训练中，已处理 Batch 数：", train_steps)
    #         avg_train_loss = total_train_loss / tf.cast(train_steps, tf.float32)
    #         print("  训练完成，总 Batch 数：", train_steps, "，平均训练 Loss：", avg_train_loss)
    #     # 验证循环同理...
    #         return



    # run_epochs_debug(epochs)

        # 使用普通的 Python 循环控制 epoch，以便调用 model.save 进行保存

    #best_val_loss = float('inf')
    best_val_loss = tf.constant(0.0101)
    print("【训练开始】共", epochs, "个 epoch。")
    for epoch in range(epochs):
        print("\n【Epoch】", epoch+1, "/", epochs, ": 开始训练...")
        total_train_loss = 0.0
        train_steps = 0
        # 训练循环
        for inputs, targets in train_dist_dataset:
            loss = distributed_train_step(inputs, targets)
            total_train_loss += float(loss)  # 转换为 float 数值
            train_steps += 1
            # print("  训练中，已处理 Batch 数：", train_steps)
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0.0
        print("  训练完成，总 Batch 数：", train_steps, "，平均训练 Loss：", avg_train_loss)
        total_val_loss = 0.0
        val_steps = 0
        # 验证循环
        for inputs, targets in val_dist_dataset:
            loss = distributed_val_step(inputs, targets)
            total_val_loss += float(loss)
            val_steps += 1
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0.0
        print("  验证完成，总 Batch 数：", val_steps, "，平均验证 Loss：", avg_val_loss)
        print("【Epoch】", epoch+1, "结束：Train Loss =", avg_train_loss, ", Val Loss =", avg_val_loss)

        # 如果验证 loss 改善，则更新并保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Validation loss improved, saving model to", best_model_path)
            model.save(best_model_path)





    # 定义训练循环，使用 tf.range 迭代 epoch，并在每个 epoch 内输出详细日志

    # @tf.function

    # def run_epochs(epochs):
    #     # 在 tf.function 外部动态创建的变量在图中只能被创建一次，
    #     # 因此这里使用 tf.Variable 来累加训练损失和步数。
    #     for epoch in tf.range(epochs):
    #         tf.print("\n【Epoch】", epoch + 1, "/", epochs, ": 开始训练...")
    #         # 重置累加变量
    #         total_train_loss = tf.zeros( [],dtype=tf.dtypes.float32)
    #         train_steps = tf.zeros( [],dtype=tf.dtypes.int32)
    #         tf.print('train_steps:',train_steps)
    #         # 训练循环
    #         for inputs, targets in train_dataset.take(-1):
    #             loss = distributed_train_step(inputs, targets)
    #             tf.print('loss:',loss)
    #             total_train_loss+= loss
    #             tf.print('total_train_loss:',total_train_loss)
    #             train_steps += 1
    #             tf.print("  训练中，已处理 Batch 数：", train_steps)
    #         avg_train_loss = total_train_loss / tf.cast(train_steps, tf.float32)
    #         tf.print("  训练完成，总 Batch 数：", train_steps, "，平均训练 Loss：", avg_train_loss)

            

    #         # 验证阶段
    #         total_val_loss = tf.zeros( [],dtype=tf.dtypes.float32)
    #         val_steps = tf.zeros( [],dtype=tf.dtypes.int32)
    #         for inputs, targets in val_dataset:
    #             loss = distributed_val_step(inputs, targets)
    #             total_val_loss += loss
    #             val_steps += 1
    #         avg_val_loss = total_val_loss / tf.cast(val_steps, tf.float32)
    #         tf.print("  验证完成，总 Batch 数：", val_steps, "，平均验证 Loss：", avg_val_loss)
    #         tf.print("【Epoch】", epoch + 1, "结束：Train Loss =", avg_train_loss, ", Val Loss =", avg_val_loss)
    #     return





    # run_epochs(tf.constant(epochs, dtype=tf.int32))  # 运行整个 epoch 循环

    return model



# -------------------------------

# 主程序：解析参数并执行训练

# -------------------------------

def main(args):
    # 使用 MirroredStrategy 利用所有可用 GPU
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices:", strategy.num_replicas_in_sync)
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    print("【数据加载】从 pickle 文件加载数据...")
    with open('X_beam.pkl', 'rb') as f:
        X1 = pickle.load(f)
    with open('Y_beam.pkl', 'rb') as f:
        Y1 = pickle.load(f)
    # 取前 4900 个样本
    X = X1[:480, :]
    y = Y1[:480, :]
    scaler = StandardScaler()
    y=scaler.fit_transform(Y1[:480,:])
    X=tf.cast(X, tf.float32)
    y=tf.cast(y, tf.float32)

    # print("X shape:", X)

    # print("y shape:", y)

    # print("X1 sample:", X1[0])

    # print("Y1 sample:", Y1[0])
    print("【数据加载】数据加载完成。")


    # 训练模型（使用自定义分布式训练循环）
    model = train_model(X, y, args.epochs, args.batch_size, args.learning_rate, args.validation_split, strategy,args.model_save_path)

    # 保存训练好的模型

    # model.save(args.model_save_path)
    print("Model saved to", args.model_save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Model Training with Multi-GPU")
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use (不适用于 MirroredStrategy)')
    parser.add_argument('--epochs', type=int, default=200, help='训练的轮数')
    parser.add_argument('--batch_size', type=int, default=20, help='每个 batch 的大小')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--validation_split', type=float, default=0.1, help='验证集所占比例')
    parser.add_argument('--model_save_path', type=str, default='./saved_model.keras', help='模型保存路径')
    args = parser.parse_args()
    main(args)

