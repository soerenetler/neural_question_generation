import tensorflow as tf
import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(self, pre_embedding=None, vocab_size=34004, embedding_dim=300, embedding_trainable=True, enc_type='bi',
                 num_layer=1, hidden_size=512,
                 cell_type='lstm', dropout=0.1, batch_sz=64):

        super(Encoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.enc_type = enc_type
        self.pre_embedding = pre_embedding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable
        self.batch_sz = batch_sz

        # Embedding Layer
        if self.pre_embedding == None:
            self.embd_layer = tf.keras.layers.Embedding(self.vocab_size,
                                                        self.embedding_dim,
                                                        trainable=True)
        else:
            embedding_matrix = np.load(self.pre_embedding)
            print("embedding_matrix:", embedding_matrix.shape)
            init = tf.keras.initializers.Constant(embedding_matrix)
            self.embd_layer = tf.keras.layers.Embedding(embedding_matrix.shape[0],
                                                        self.embedding_dim,
                                                        embeddings_initializer=init,
                                                        trainable=self.embedding_trainable)
        enc_cell = self._create_cell()
        print("get_initial_state:", enc_cell.get_initial_state(
            batch_size=64, dtype=tf.float32))
        # Encoder
        if self.enc_type == 'mono':
            # encoder_output: [batch_size, max_time, hidden_size]
            # encoder_state: last hidden_state of encoder, [batch_size, hidden_size]
            print("enc_cell: ", enc_cell)
            self.rnn = tf.keras.layers.RNN(enc_cell, return_sequences=True,
                                           return_state=True)

        elif self.enc_type == 'bi':
            print(enc_cell)
            self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(
                enc_cell, return_sequences=True, 
                return_state=True))

        else:
            raise ValueError('Invalid input %s' % self.enc_type)

    def call(self, inputs, hidden, training=False):
        embd_inputs = self.embd_layer(inputs)
        print("embd_inputs:", embd_inputs.shape)
        print("initial_state:", hidden)
        print("rnn: ", self.rnn)
        result_encoder = self.rnn(
            embd_inputs, initial_state=hidden, training=training)
        print("result_encoder:", result_encoder)
        if self.enc_type == 'mono':
            if self.cell_type == 'gru':
                encoder_output, encoder_state = result_encoder
            else:  # lstm
                encoder_output, encoder_state_h, encoder_state_c = result_encoder
                encoder_state = [encoder_state_h, encoder_state_c]

        elif self.enc_type == 'bi':
            if self.cell_type == 'gru':
                encoder_output = tf.concat(encoder_output, -1)
                if self.num_layer == 1:
                    encoder_state = tf.concat(encoder_state, -1)
                else:  # multi layer
                    encoder_state = tuple(tf.concat(
                        [state_fw, state_bw], -1) for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]))
            else:  # lstm
                print("result_encoder ", len(result_encoder))
                encoder_output, encoder_state_h, encoder_state_c = result_encoder
                encoder_output = tf.concat(encoder_output, -1)
                if self.num_layer == 1:
                    encoder_state_c = tf.concat(
                        [encoder_state[0].c, encoder_state[1].c], 1)
                    encoder_state_h = tf.concat(
                        [encoder_state[0].h, encoder_state[1].h], 1)
                    encoder_state = dict(c=encoder_state_c, h=encoder_state_h)
                else:  # multi layer
                    _encoder_state = list()
                    for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]):
                        partial_state_c = tf.concat(
                            [state_fw.c, state_bw.c], 1)
                        partial_state_h = tf.concat(
                            [state_fw.h, state_bw.h], 1)
                        partial_state = dict(
                            c=partial_state_c, h=partial_state_h)
                        _encoder_state.append(partial_state)
                    encoder_state = tuple(_encoder_state)

        return encoder_output, encoder_state

    # Build cell for encoder and decoder
    def _create_cell(self):
        rnn_cell = tf.keras.layers.GRUCell(
            self.hidden_size, dropout=self.dropout) if self.cell_type == 'gru' else tf.keras.layers.LSTMCell(self.hidden_size, dropout=self.dropout)
        return rnn_cell if self.num_layer == 1 else tf.keras.layers.StackedRNNCells(([rnn_cell for _ in range(self.num_layer)]))

    def initialize_hidden_state(self):
        if self.enc_type == 'mono':
            if self.cell_type == 'gru':
                pass
            else:  # LSTM
                return [tf.zeros((self.batch_sz, self.hidden_size)), tf.zeros((self.batch_sz, self.hidden_size))]
        else: # bi
            if self.cell_type == 'gru':
                pass
            else:  # LSTM
                return [tf.zeros((self.batch_sz, self.hidden_size)), tf.zeros((self.batch_sz, self.hidden_size))] * 2
