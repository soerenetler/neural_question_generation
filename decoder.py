import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Decoder(tf.keras.layers.Layer):
    def __init__(self, pre_embedding=None, vocab_size=34004, embedding_dim=300, enc_type='bi',
                 attn_type='bahdanau', voca_size=None,
                 beam_width=0, length_penalty_weight=1,
                 num_layer=1, hidden_size=512,
                 cell_type='lstm', dropout=0.1, batch_sz=64, max_length_input=32, embedding_trainable= False,
                 sample_prob=0.25):

        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.enc_type = enc_type
        self.attn_type = attn_type
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.voca_size = voca_size
        self.sample_prob = sample_prob
        self.pre_embedding = pre_embedding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable
        self.batch_sz = batch_sz
        self.max_length_input=max_length_input
        self.max_length_output=max_length_input

        if self.pre_embedding == None:
            self.embd_layer = tf.keras.layers.Embedding(self.vocab_size,
                                                        self.embedding_dim,
                                                        trainable=True)
        else:
            embedding_matrix = np.load(self.pre_embedding)
            init = tf.keras.initializers.Constant(embedding_matrix)
            self.embd_layer = tf.keras.layers.Embedding(self.vocab_size,
                                                        self.embedding_dim,
                                                        embeddings_initializer=init,
                                                        trainable=self.embedding_trainable)

        print("hidden_size:", self.hidden_size)
        
        # Define the fundamental cell for decoder recurrent structure
        self.dec_cell = self._create_cell()

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(self.vocab_size)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        #else:  # EVAL & TEST
        #   self.sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self._attention(None, self.batch_sz*[self.max_length_input])

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.dec_cell,
                                                     self.attention_mechanism, attention_layer_size=self.hidden_size)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def call(self, dec_input, decoder_initial_state, start_token=1, end_token=2, mode=tf.estimator.ModeKeys.TRAIN):
        # batch_size should not be specified
        # if fixed, then the redundant evaluation data will make error
        # it may related to bug of tensorflow api

        embd_input = self.embd_layer(dec_input)
        outputs, _, _ = self.decoder(embd_input, initial_state=decoder_initial_state, sequence_length=self.batch_sz*[self.max_length_output-1])
        return outputs

        # Decoder initial state setting
        # if (self.mode != tf.estimator.ModeKeys.PREDICT or self.beam_width == 0):
        #     initial_state = self.rnn_cell.get_initial_state(batch_size=self.batch_size, dtype=self.dtype)
        #     initial_state = initial_state.clone(cell_state=encoder_state)

        # else:
        #     initial_state = self.rnn_cell.get_initial_state(batch_size=self.batch_size * self.beam_width, dtype=self.dtype)
        #     initial_state = initial_state.clone(cell_state=encoder_state)
        #     print(type(self.length_penalty_weight))
        #     print('----------------------------------')
        #     decoder = tfa.seq2seq.BeamSearchDecoder(
        #         cell=self.rnn_cell,
        #         embedding=embedding,
        #         start_tokens=start_token,
        #         end_token=end_token,
        #         initial_state=initial_state,
        #         beam_width=self.beam_width,
        #         length_penalty_weight=self.length_penalty_weight)

        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     outputs, _, _ = tfa.seq2seq.dynamic_decode(
        #         decoder, impute_finished=True, maximum_iterations=None)
        #     return outputs.rnn_output

        # # Test with Beam decoding
        # elif (mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0):
        #     outputs, _, _ = tfa.seq2seq.dynamic_decode(
        #         decoder, impute_finished=False, maximum_iterations=self.max_iter)
        #     predictions = outputs.predicted_ids  # [batch, length, beam_width]
        #     # [batch, beam_width, length]
        #     predictions = tf.transpose(a=predictions, perm=[0, 2, 1])
        #     predictions = predictions[:, 0, :]  # [batch, length]
        #     return predictions

        # else:  # Greedy decoder (Test & Eval)
        #     outputs, _, _ = tfa.seq2seq.dynamic_decode(
        #         decoder, impute_finished=True, maximum_iterations=self. max_iter)
        #     return outputs.rnn_output

    # def set_attention_cell(self, memory, memory_length, encoder_state, enc_num_layer):
    #     self.batch_size = tf.shape(input=memory)[0]

    #     dec_cell = self._create_cell()
    #     copy_state = encoder_state

    #     if (mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0):
    #         memory = tfa.seq2seq.tile_batch(memory, self.beam_width)
    #         memory_length = tfa.seq2seq.tile_batch(
    #             memory_length, self.beam_width)
    #         copy_state = tfa.seq2seq.tile_batch(
    #             copy_state, self.beam_width)

    #     attention_mechanism = self._attention(memory, memory_length)

    #     initial_cell_state = copy_state if self.num_layer == enc_num_layer else None
    #     attn_dec_cell = tfa.seq2seq.AttentionWrapper(
    #         dec_cell, attention_mechanism,
    #         attention_layer_size=self.hidden_size,
    #         initial_cell_state=initial_cell_state)

    #     # Set maximum iteration for GreedyHelper(Eval and Test)
    #     self.max_iter = None if mode == tf.estimator.ModeKeys.TRAIN else tf.round(
    #         tf.reduce_max(input_tensor=memory_length) * 2)

    def _attention(self, memory, memory_length):
        if self.attn_type == 'bahdanau':
            return tfa.seq2seq.BahdanauAttention(
                units=self.hidden_size,
                memory=memory,
                memory_sequence_length=memory_length)
        elif self.attn_type == 'normed_bahdanau':
            return tfa.seq2seq.BahdanauAttention(
                units=self.hidden_size,
                memory=memory,
                memory_sequence_length=memory_length,
                normalize=True)

        elif self.attn_type == 'luong':
            return tfa.seq2seq.LuongAttention(
                units=self.hidden_size,
                memory=memory,
                memory_sequence_length=memory_length)

        elif self.attn_type == 'scaled_luong':
            return tfa.seq2seq.LuongAttention(
                units=self.hidden_size,
                memory=memory,
                memory_sequence_length=memory_length,
                scale=True)
        else:
            raise ValueError('Unknown attention mechanism : %s' %
                             self.attn_type)

    # Build cell for encoder and decoder
    def _create_cell(self):
        rnn_cell = tf.keras.layers.GRUCell(
                self.hidden_size, dropout=self.dropout) if self.cell_type == 'gru' else tf.keras.layers.LSTMCell(self.hidden_size, dropout=self.dropout)
        return rnn_cell if self.num_layer == 1 else tf.keras.layers.StackedRNNCells(([rnn_cell for _ in range(self.num_layer)]))

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        print("decoder_initial_state: ", decoder_initial_state)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

