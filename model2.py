import tensorflow as tf

def create_node_embedding(embedding_dim):
    """
    Create a node embedding sub-model that projects 7-dimensional input
    (2 coordinates + 3global parameters) into a higher-dimensional space.
    """
    input_features = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(embedding_dim, activation='relu')(input_features)
    return tf.keras.models.Model(inputs=input_features, outputs=x)

def transformer_encoder_layer(embedding_dim, num_heads, ff_dim, dropout_rate):
    """
    Build a single transformer encoder layer with multi-head attention and a feed-forward network.
    """
    inputs = tf.keras.layers.Input(shape=(None, embedding_dim))
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(embedding_dim)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return tf.keras.models.Model(inputs=inputs, outputs=out2)

def build_transformer_model (embedding_dim=128, num_heads=2, ff_dim=256, num_layers=4, dropout_rate=0.2):
    """
    Build the complete transformer model:
      - Each sample is of shape (num_nodes, 4)
      - A TimeDistributed node embedding converts each node to a 128-dimensional vector
      - Stacked transformer encoder layers process these embeddings
      - Global average pooling aggregates the tokens
      - A final Dense layer outputs 10 values (LVV time points)
    """
    inputs = tf.keras.layers.Input(shape=(None, 4))
    node_embedding_model = create_node_embedding(embedding_dim)
    embedded_nodes = tf.keras.layers.TimeDistributed(node_embedding_model)(inputs)
    x = embedded_nodes
    for _ in range(num_layers):
        encoder_layer = transformer_encoder_layer(embedding_dim, num_heads, ff_dim, dropout_rate)
        x = encoder_layer(x)
    
    #x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(20),
        name="per_token_output"
    )(x)    
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
