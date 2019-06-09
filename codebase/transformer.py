import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_datasets as tfds
import numpy as np

import time
import matplotlib.pyplot as plt


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

###
pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
################################################################################

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones([size, size]), -1, 0)
    return mask

def create_masks(x, y):
    encoder_padding_mask = create_padding_mask(x)
    decoder_padding_mask = create_padding_mask(x)
    look_ahead_mask = create_look_ahead_mask(y.shape[1])
    decoder_padding_mask2 = create_padding_mask(y)
    combined_mask = tf.maximum(decoder_padding_mask2, look_ahead_mask)

    return encoder_padding_mask, combined_mask, decoder_padding_mask

###
x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)

create_look_ahead_mask(5)
################################################################################

def scaled_dot_product_attention(Q, K, V, mask):
    """
    Input:
    Q (batch, num_heads, seq_len1, dk): query
    K (batch, num_heads, seq_len2, dk): keys
    V (batch, num_heads, seq_len2, dv): values

    The dimension of batch, num_heads can be empty.

    mask (seq_len1, seq_len2)

    Output:
    output (batch, num_heads, seq_len1, dv)
    attention_weights (batch, num_heads, seq_len1, seq_len2)
    """
    ## (batch, num_heads, seq_len1, seq_len2)
    QK = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(K.shape[-1], tf.float32)
    ## (batch, num_heads, seq_len1, seq_len2)
    scaled_attention_logits = QK / tf.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    ## (batch, num_heads, seq_len1, seq_len2)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    ## (batch, num_heads, seq_len1, dv)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights

###
Q = tf.random.uniform([1, 2, 10])
K = tf.random.uniform([1, 3, 10])
V = tf.random.uniform([1, 3, 5])

output, attention_weights = scaled_dot_product_attention(Q, K, V, None)
################################################################################
class MultiHeadAttention(tfk.layers.Layer):
    def __init__(self, model_depth, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_depth = model_depth

        assert self.model_depth % self.num_heads == 0

        self.depth = model_depth // self.num_heads

        self.q2q = tfk.layers.Dense(self.model_depth)
        self.k2k = tfk.layers.Dense(self.model_depth)
        self.v2v = tfk.layers.Dense(self.model_depth)

        self.o2o = tfk.layers.Dense(self.model_depth)

    def split_heads(self, x):
        """
        Input:
        x (batch, seq_len, model_depth)

        Output:
        o (batch, num_heads, seq_len, depth)
        """
        o = tf.reshape(x, [x.shape[0], -1, self.num_heads, self.depth])
        return tf.transpose(o, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        """
        Input:
        q (batch, seq_len1, embedding_size_q)
        k (batch, seq_len2, embedding_size_k)
        v (batch, seq_len2, embedding_size_v)
        mask (seq_len1, seq_len2)

        Output:
        o (batch, seq_len1, model_depth)
        attention_weigths (batch, num_heads, seq_len1, seq_len2)
        """
        q = self.q2q(q) # (batch, seq_len1, model_depth)
        k = self.k2k(k) # (batch, seq_len2, model_depth)
        v = self.v2v(v) # (batch, seq_len2, model_depth)

        q = self.split_heads(q) # (batch, num_heads, seq_len1, depth)
        k = self.split_heads(k) # (batch, num_heads, seq_len2, depth)
        v = self.split_heads(v) # (batch, num_heads, seq_len2, depth)

        ## (batch, num_heads, seq_len1, depth), (batch, num_heads, seq_len1, seq_len2)
        o, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        ## (batch, seq_len1, num_heads, depth)
        o = tf.transpose(o, perm=[0, 2, 1, 3])
        ## (batch, seq_len1, model_depth)
        o = tf.reshape(o, [o.shape[0], -1, self.model_depth])
        o = self.o2o(o)
        return o, attention_weights

###
mha = MultiHeadAttention(model_depth=512, num_heads=8)
x = tf.random.uniform([1, 60, 512]) # (batch, seq_len, model_depth)
out, attn = mha(x, x, x, mask=None)
print(out.shape)
print(attn.shape)

y = tf.random.uniform([1, 60, 512])
x = tf.random.uniform([1, 30, 512])
out, attn = mha(y, x, x, mask=None)
print(out.shape)
print(attn.shape)
################################################################################
def pointwise_feed_forward_network(model_depth, hidden_dim):
    return tfk.Sequential([
        tfk.layers.Dense(hidden_dim, activation='relu'),
        tfk.layers.Dense(model_depth)
    ])

###
ffn = pointwise_feed_forward_network(512, 2048)
x = tf.random.uniform([64, 50, 256]) # (batch, seq_len, *)
ffn(x).shape
################################################################################
class EncoderLayer(tfk.layers.Layer):
    def __init__(self, model_depth, num_heads, ffn_hidden_size, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(model_depth, num_heads)
        self.ffn = pointwise_feed_forward_network(model_depth, ffn_hidden_size)

        self.layernorm1 = tfk.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tfk.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tfk.layers.Dropout(dropout_rate)
        self.dropout2 = tfk.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Input:
        x (batch, seq_len2, model_depth)
        training: True or False
        mask (seq_len2, seq_len2)

        Output:
        o2 (batch, seq_len2, model_depth)
        """
        ## (batch, seq_len2, model_depth)
        o1, _ = self.mha(x, x, x, mask)
        o1 = self.dropout1(o1, training=training)
        o1 = self.layernorm1(x + o1)

        ## (batch, seq_len2, model_depth)
        o2 = self.ffn(o1)
        o2 = self.dropout2(o2, training=training)
        o2 = self.layernorm2(o1 + o2)

        return o2

class Encoder(tfk.layers.Layer):
    def __init__(self, num_layers, model_depth, num_heads, ffn_hidden_size,
                 input_vocab_size, dropout_rate=0.1, max_length=MAX_LENGTH):
        super(Encoder, self).__init__()

        self.model_depth = model_depth
        self.num_layers = num_layers

        self.embedding = tfk.layers.Embedding(input_vocab_size, self.model_depth)
        self.pos_encoding = positional_encoding(max_length+5, self.model_depth)

        self.encoder_layers = [EncoderLayer(model_depth, num_heads, ffn_hidden_size,
                                            dropout_rate) for _ in range(num_layers)]
        self.dropout = tfk.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Input:
        x (batch, seq_len2)
        training: True or False
        mask (seq_len2, seq_len2)

        Output:
        o (batch, seq_len2, model_depth)
        """
        seq_len = x.shape[1]

        o = self.embedding(x)
        o = o * tf.sqrt(self.model_depth * 1.0)
        o = o + self.pos_encoding[:, :seq_len, :]

        o = self.dropout(o, training=training)

        for i in range(self.num_layers):
            o = self.encoder_layers[i](o, training, mask)

        return o
###
encoder_layer = EncoderLayer(512, 8, 2048)
encoder_layer_output = encoder_layer(tf.random.uniform([64, 43, 512]), False, None)
encoder_layer_output.shape

encoder = Encoder(num_layers=2, model_depth=512, num_heads=8,
                  ffn_hidden_size=2048, input_vocab_size=8500)
encoder_output = encoder(tf.random.uniform([64, 30]), training=False,
                         mask=None)
encoder_output.shape
################################################################################
class DecoderLayer(tfk.layers.Layer):
    def __init__(self, model_depth, num_heads, ffn_hidden_size, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(model_depth, num_heads)
        self.mha2 = MultiHeadAttention(model_depth, num_heads)

        self.ffn = pointwise_feed_forward_network(model_depth, ffn_hidden_size)

        self.layernorm1 = tfk.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tfk.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tfk.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tfk.layers.Dropout(dropout_rate)
        self.dropout2 = tfk.layers.Dropout(dropout_rate)
        self.dropout3 = tfk.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Input:
        x (batch, seq_len1, model_depth): target
        encoder_output (batch, seq_len2, model_depth): source
        look_ahead_mask (seq_len1, seq_len1): target mask
        padding_mask (seq_len1, seq_len2): source mask

        Output:
        o3 (batch, seq_len1, model_depth)
        attn_weights1 (batch, num_heads, seq_len1, seq_len1)
        attn_weights2 (batch, num_heads, seq_len1, seq_len2)
        """
        ## o1 (batch, seq_len1, model_depth)
        o1, attn_weights1 = self.mha1(x, x, x, look_ahead_mask)
        o1 = self.dropout1(o1, training=training)
        o1 = self.layernorm1(o1 + x)

        ## o2 (batch, seq_len1, model_depth)
        o2, attn_weights2 = self.mha2(o1, encoder_output, encoder_output, padding_mask)
        o2 = self.dropout2(o2, training=training)
        o2 = self.layernorm2(o2 + o1)

        ## o3 (batch, seq_len1, model_depth)
        o3 = self.ffn(o2)
        o3 = self.dropout3(o3, training=training)
        o3 = self.layernorm3(o3 + o2)

        return o3, attn_weights1, attn_weights2

class Decoder(tfk.layers.Layer):
    def __init__(self, num_layers, model_depth, num_heads, ffn_hidden_size,
                 target_vocab_size, dropout_rate=0.1, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()

        self.model_depth = model_depth
        self.num_layers = num_layers

        self.embedding = tfk.layers.Embedding(target_vocab_size, self.model_depth)
        self.pos_encoding = positional_encoding(max_length+5, self.model_depth)

        self.decoder_layers = [DecoderLayer(model_depth, num_heads, ffn_hidden_size,
                                            dropout_rate) for _ in range(num_layers)]
        self.dropout = tfk.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        Input:
        x (batch, seq_len1)
        encoder_output (batch, seq_len2, model_depth)
        training: True or False
        look_ahead_mask (seq_len1, seq_len1)
        padding_mask (seq_len1, seq_len2)

        o (batch, seq_len1, model_depth)
        """

        seq_len = x.shape[1]
        attention_weights = {}

        ## (batch, seq_len1, model_depth)
        o = self.embedding(x)
        o = o * tf.sqrt(tf.cast(self.model_depth, tf.float32))
        o = o + self.pos_encoding[:, :seq_len, :]
        o = self.dropout(o, training=training)

        for i in range(self.num_layers):
            o, block1, block2 = self.decoder_layers[i](o, encoder_output, training,
                                                       look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return o, attention_weights

###
decoder_layer = DecoderLayer(512, 8, 2048)
decoder_layer_output, _, _ = decoder_layer(tf.random.uniform([64, 50, 512]), encoder_output,
    False, None, None)
decoder_layer_output.shape

decoder = Decoder(num_layers=2, model_depth=512, num_heads=8,
                  ffn_hidden_size=2048, target_vocab_size=8000)
print(encoder_output.shape)
decoder_output, attn = decoder(tf.random.uniform([64, 26]),
                               encoder_output=encoder_output,
                               training=False, look_ahead_mask=None, padding_mask=None)
print(decoder_output.shape)
################################################################################
class Transformer(tfk.Model):
    def __init__(self, num_layers, model_depth, num_heads, ffn_hidden_size,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, model_depth, num_heads, ffn_hidden_size,
                               input_vocab_size, dropout_rate)
        self.decoder = Decoder(num_layers, model_depth, num_heads, ffn_hidden_size,
                               target_vocab_size, dropout_rate)
        self.final_layer = tfk.layers.Dense(target_vocab_size)

    def call(self, x, y, training, encoder_padding_mask, look_ahead_mask,
             decoder_padding_mask):
        encoder_output = self.encoder(x, training, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(y, encoder_output,
            training, look_ahead_mask, decoder_padding_mask)
        o = self.final_layer(decoder_output)

        return o, attention_weights

###
transformer = Transformer(num_layers=2, model_depth=512, num_heads=8, ffn_hidden_size=2048,
                          input_vocab_size=8500, target_vocab_size=8000)

x = tf.random.uniform([64, 32], maxval=8500, dtype=tf.int32)
y = tf.random.uniform([64, 26], maxval=8000, dtype=tf.int32)

o, _ = transformer(x, y, training=False,
                   encoder_padding_mask=None,
                   look_ahead_mask=None,
                   decoder_padding_mask=None)
o.shape
################################################################################
def evaluate (input_sentence):
    """
    input_sentence is a sentence to the encoder.
    """
    x = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(input_sentence)\
        + [tokenizer_pt.vocab_size + 1]
    x = tf.expand_dims(x, 0) # add batch dimension

    y = tf.expand_dims([tokenizer_en.vocab_size], 0) # (batch, seq_len)

    for i in range(MAX_LENGTH):
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = create_masks(x, y)

        predictions, attention_weights = transformer(x,
                                                     y,
                                                     False,
                                                     encoder_padding_mask,
                                                     look_ahead_mask,
                                                     decoder_padding_mask)
        ## (batch, 1, vocab_size)
        last_prediction = predictions[:, -1:, :]
        ## (batch, 1)
        last_token = tf.cast(tf.argmax(last_prediction, axis=-1), tf.int32)

        ## reached the end token
        if tf.equal(last_token, tokenizer_en.vocab_size+1):
            return tf.squeeze(y, axis=0), attention_weights

        ## concatenate the last token to y
        y = tf.concat([y, last_token], axis=-1)

    return tf.squeeze(y, axis=0), attention_weights
