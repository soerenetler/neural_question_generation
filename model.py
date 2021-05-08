import sys

import tensorflow as tf
import tensorflow_addons as tfa

from encoder import Encoder
from decoder import Decoder
import utils


class QG(tf.keras.Model):
    PAD = 0
    GO = 1
    EOS = 2
    UNK = 3

    def __init__(self, params):
        super(QG, self).__init__()
        self.vocab_size = params['voca_size']
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.cell_type = params['cell_type']
        self.pre_embedding = params['pre_embedding']
        self.embedding_trainable = params['embedding_trainable']
        self.enc_type = params['enc_type']
        self.num_layer = params['num_layer']
        # for loss calculation
        self.batch_size = params['batch_size']
        self.maxlen_s = params['maxlen_s']
        self.maxlen_dec_train = params['maxlen_dec_train']
        self.maxlen_dec_dev = params['maxlen_dec_dev']  # for loss calculation
        self.rnn_dropout = params['dropout']
        self.attn = params['attn']
        self.beam_width = params['beam_width']
        self.length_penalty_weight = params['length_penalty_weight']

        self.encoder = Encoder(pre_embedding=self.pre_embedding, vocab_size=self.vocab_size, embedding_dim=self.embedding_size,
                               embedding_trainable=self.embedding_trainable, enc_type=self.enc_type,
                               num_layer=self.num_layer, hidden_size=self.hidden_size,
                               cell_type=self.cell_type, dropout=self.rnn_dropout, batch_sz=self.batch_size
                               )
        if (self.enc_type == 'bi'):
            hidden_size = 2 * self.hidden_size
        else:
            hidden_size = self.hidden_size
        self.decoder = Decoder(pre_embedding=self.pre_embedding, vocab_size=self.vocab_size, embedding_dim=self.embedding_size, enc_type=self.enc_type,
                               attn_type=self.attn,
                               beam_width=self.beam_width, length_penalty_weight=self.length_penalty_weight,
                               num_layer=self.num_layer, dec_units=hidden_size,
                               cell_type=self.cell_type, dropout=self.rnn_dropout, batch_sz=self.batch_size,
                               max_length_input=self.maxlen_s, max_length_output=self.maxlen_dec_train, embedding_trainable=self.embedding_trainable)

    def call(self, inputs, training=False):
        if training:
            enc_inp, dec_input = inputs
        else:  # EVLA/PREDICT
            enc_inp = inputs
            dec_input = None

        print("CALL - model - TRAINING: ", training)
        enc_hidden = self.encoder.initialize_hidden_state()

        print("CALL - model - enc_inp.shape: ", enc_inp.shape)

        enc_output, enc_hidden = self.encoder(
            enc_inp, enc_hidden, training=training)

        print("CALL - model - enc_hidden: [batch_size, hidden_size]", enc_hidden.shape)
        print("CALL - model - enc_output [batch_size, max_time, hidden_size]: ", enc_output.shape)

        if training:
            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)
        else:
            # From official documentation
            # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
            # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
            # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
            # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.
            enc_out = tfa.seq2seq.tile_batch(
                enc_output, multiplier=self.beam_width)
            print("CALL - model - enc_out.shape = beam_with * [batch_size, max_length_input, rnn_units] :", enc_out.shape)
            self.decoder.attention_mechanism.setup_memory(enc_out)

        # Create AttentionWrapperState as initial_state for decoder
        pred = self.decoder(dec_input, enc_hidden, start_token=self.GO, end_token=self.EOS, training=training)

        return pred

    @tf.function
    def train_step(self, data):
        encoder_inp, targ = data
        loss = 0

        with tf.GradientTape() as tape:
            dec_input = targ[:, :-1]  # Ignore <end> token
            real = targ[:, 1:]         # ignore <start> token
            pred = self((encoder_inp, dec_input),
                        training=True)  # Forward pass
            print("train_step - pred: ", pred)
            logits = pred.rnn_output
            loss = self.loss(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {"loss": loss}

    #     # Add attention wrapper to decoder cell
    #     self.decoder.set_attention_cell(
    #         encoder_outputs, enc_input_length, encoder_state, self.enc_layer)

    #     if not (mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0):
    #         logits = self.decoder.call(
    #             dec_inputs, self.dec_input_length, self.GO, self.EOS)
    #         predictions = tf.argmax(input=self.logits, axis=-1)

    #     else:  # Beam decoding
    #         predictions = self.decoder.call(
    #             dec_inputs, dec_input_length, self.GO, self.EOS)

    #     self._calculate_loss(predictions, labels, mode)
    #     return self._update_or_output(mode)

    # def _calculate_length(self, inputs):
    #     input_length = tf.reduce_sum(
    #         input_tensor=tf.cast(tf.not_equal(inputs, self.PAD), tf.int32), axis=-1)
    #     return input_length

    # def _calculate_loss(self, dec_inputs,  mode):
    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         return

    #     self.labels = tf.concat([dec_inputs[:, 1:], tf.zeros(
    #         [self.batch_size, 1], dtype=tf.int32)], 1, name='labels')
    #     maxlen_label = self.maxlen_dec_train if mode == tf.estimator.ModeKeys.TRAIN else self.maxlen_dec_dev
    #     current_length = tf.shape(input=self.logits)[1]

    #     def concat_padding():
    #         num_pad = maxlen_label - current_length
    #         padding = tf.zeros(
    #             [self.batch_size, num_pad, self.voca_size], dtype=self.dtype)
    #         return tf.concat([self.logits, padding], 1)

    #     def slice_to_maxlen():
    #         return tf.slice(self.logits, [0, 0, 0], [self.batch_size, maxlen_label, self.voca_size])

    #     self.logits = tf.cond(pred=current_length < maxlen_label,
    #                           true_fn=concat_padding,
    #                           false_fn=slice_to_maxlen)

    #     weight_pad = tf.sequence_mask(
    #         self.dec_input_length, maxlen_label, self.dtype)
    #     self.loss = tfa.seq2seq.sequence_loss(
    #         self.logits,
    #         self.labels,
    #         weight_pad,
    #         average_across_timesteps=True,
    #         average_across_batch=True,
    #         softmax_loss_function=None  # default : sparse_softmax_cross_entropy
    #     )





 