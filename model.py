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
        self.voca_size = params['voca_size']
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.cell_type = params['cell_type']
        self.pre_embedding = params['pre_embedding']
        self.embedding_trainable = params['embedding_trainable']
        self.enc_type = params['enc_type']
        self.enc_layer = params['encoder_layer']
        self.dec_layer = params['decoder_layer']
        # for loss calculation
        self.maxlen_dec_train = params['maxlen_dec_train']
        self.maxlen_dec_dev = params['maxlen_dec_dev']  # for loss calculation
        self.rnn_dropout = params['dropout']
        self.attn = params['attn']
        self.beam_width = params['beam_width']
        self.length_penalty_weight = params['length_penalty_weight']
        self.sample_prob = params['sample_prob']
        self.learning_rate = params['learning_rate']
        self.decay_step = params['decay_step']  # learning rate decay
        self.decay_rate = params['decay_rate']  # learning rate decay step

        self.encoder = Encoder(pre_embedding=self.pre_embedding, vocab_size=self.voca_size, embedding_dim=self.embedding_size,
                               embedding_trainable=self.embedding_trainable, enc_type=self.enc_type,
                               num_layer=self.enc_layer, hidden_size=self.hidden_size,
                               cell_type=self.cell_type, dropout=self.rnn_dropout, batch_sz=64
                               )
        if (self.enc_type == 'bi'):
            hidden_size = 2 * self.hidden_size
        else:
            hidden_size = self.hidden_size
        self.decoder = Decoder(enc_type=self.enc_type,
                               attn_type=self.attn, voca_size=self.voca_size,
                               beam_width=self.beam_width, length_penalty_weight=self.length_penalty_weight,
                               num_layer=self.dec_layer, hidden_size=hidden_size,
                               cell_type=self.cell_type, dropout=self.rnn_dropout,
                               sample_prob=self.sample_prob, batch_sz=64, max_length_input=32, embedding_trainable= False,)

    def call(self, inputs, training=False):
        enc_inp, dec_input = inputs
        self.batch_size = tf.shape(input=enc_inp)[0]
        enc_hidden = self.encoder.initialize_hidden_state()

        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)

        # Set the AttentionMechanism object with encoder_outputs
        self.decoder.attention_mechanism.setup_memory(enc_output)


        print("enc_hidden: ", enc_hidden)
        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = self.decoder.build_initial_state(
            self.batch_size, enc_hidden, tf.float32)
        pred = self.decoder(dec_input, decoder_initial_state)

        return pred

    def compile(self, optimizer, loss_fn):
        super(QG, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data):
        encoder_inp = data["enc_inputs"]
        targ  = data["dec_inputs"]
        loss = 0

        with tf.GradientTape() as tape:
            dec_input = targ[:, :-1]  # Ignore <end> token
            real = targ[:, 1:]         # ignore <start> token
            pred = self((encoder_inp, dec_input), training=True)  # Forward pass

            logits = pred.rnn_output
            loss = self.loss_fn(real, logits)

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

    # def _update_or_output(self, predictions, labels, mode):

    #     if mode == tf.estimator.ModeKeys.PREDICT:
    #         return tf.estimator.EstimatorSpec(
    #             mode=mode,
    #             predictions={
    #                 'question': predictions
    #             })
    #     eval_metric_ops = {
    #         'bleu': utils.bleu_score(labels, self.predictions)
    #     }

    #     # Optimizer
    #     if self.decay_step is not None:
    #         self.learning_rate = tf.compat.v1.train.exponential_decay(
    #             self.learning_rate,
    #             tf.compat.v1.train.get_global_step(),
    #             self.decay_step,
    #             self.decay_rate,
    #             staircase=True)

    #     optimizer = tf.optimizers.Adam(self.learning_rate)

    #     grad_and_var = optimizer.compute_gradients(
    #         self.loss, tf.compat.v1.trainable_variables())
    #     grad, var = zip(*grad_and_var)
    #     # grad, norm = tf.clip_by_global_norm(grad, 5)
    #     train_op = optimizer.apply_gradients(
    #         zip(grad, var), global_step=tf.compat.v1.train.get_global_step())

    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         loss=self.loss,
    #         train_op=train_op,
    #         eval_metric_ops=eval_metric_ops)
