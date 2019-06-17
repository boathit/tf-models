import tensorflow as tf
import tensorflow.keras as tfk

import matplotlib.pyplot as plt

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tfk.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tfk.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')

    def call(self, x, h0):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=h0)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros([self.batch_size, self.enc_units])

###
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

h0 = encoder.initialize_hidden_state()
sample_output, hn = encoder(example_input_batch, h0)

print ('Encoder output shape: (batch size, sequence length, units) {}'.\
    format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(hn.shape))

################################################################################
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tfk.layers.Dense(units)
        self.W2 = tfk.layers.Dense(units)
        self.V  = tfk.layers.Dense(1)

    def call(self, query, values):
        """
        query (batch size, hidden size): h_t
        values (batch size, max length, hidden size): h_s
        """
        ## (batch size, hidden size) => (batch size, 1, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)
        ## (batch size, max length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        ## normalizing along the second dimension
        ## (batch size, max length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        ## (batch size, max length, hidden size) => (batch size, hidden size)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)

        return context_vector, attention_weights

###
attention_layer = BahdanauAttention(10)
context_vector, attention_weights = attention_layer(hn, sample_output)

print("Attention result shape: (batch size, units) {}".format(context_vector.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".\
    format(attention_weights.shape))

################################################################################
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tfk.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tfk.layers.GRU(self.dec_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
        self.fc = tfk.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        """
        x (batch size, 1)
        hidden (batch size, hidden size)
        enc_output (batch size, max length, hidden size)
        """
        ## (batch size, hidden size), (batch size, hidden size, 1)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        ## (batch size, 1, embedding size)
        x = self.embedding(x)

        ## (batch size, 1, hidden size + embedding size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        ## (batch size, 1, dec_units), (batch size, dec_units)
        output, state = self.gru(x)

        ## (batch size, dec_units)
        output = tf.reshape(output, [-1, output.shape[-1]])

        ## (batch size, vocab size)
        x = self.fc(output)

        return x, state, attention_weights

###
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform([64, 1]),
                                      hn, sample_output)
print ('Decoder output shape: (batch_size, vocab size) {}'.\
    format(sample_decoder_output.shape))

################################################################################
def evaluate(sentence):
    attention_plot = np.zeros([max_length_targ, max_length_inp])

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[w] for w in sentence.split(' ')]
    inputs = tfk.preprocessing.sequence.pad_sequences([inputs],
                                                      maxlen=max_length_inp,
                                                      padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result= ''

    enc_h0 = tf.zeros([1, units])
    enc_output, enc_hn = encoder(inputs, enc_h0)

    dec_h0 = enc_hn
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 1)

    ## iterating over the time axis
    for i in range(max_length_targ):
        pred, dec_h0, attention_weights = decoder(dec_input, dec_h0, enc_output)

        attention_weights = tf.reshape(attention_weights, [-1, ])
        attention_plot[i] = attention_weights.numpy()

        pred_id = tf.argmax(pred[0]).numpy()

        result += targ_lang.index_word[pred_id] + ' '

        if targ_lang.index_word[pred_id] == '<end>':
            return result, sentence, attention_plot

        ## preparing decoder input for next iteration
        dec_input = tf.expand_dims([pred_id], 0)

    return result, sentence, attention_plot
