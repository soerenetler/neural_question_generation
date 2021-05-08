import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from utils import _create_cell, build_attention_mechanism


class Decoder(tf.keras.layers.Layer):
    def __init__(self, pre_embedding=None, vocab_size: int = 34004, embedding_dim: int = 300, enc_type: str = 'bi',
                 attn_type: str = 'bahdanau',
                 beam_width: int = 0, length_penalty_weight: float = 1,
                 num_layer: int = 1, dec_units: int = 512,
                 cell_type: str = 'lstm', dropout: float = 0.1, batch_sz: int = 64, max_length_input: int = 60,
                 max_length_output: int = 32, embedding_trainable: bool = False):

        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.dec_units = dec_units
        self.cell_type = cell_type
        self.dropout = dropout
        self.enc_type = enc_type
        self.attn_type = attn_type
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.pre_embedding = pre_embedding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable
        self.batch_sz = batch_sz
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        print("INIT - decoder - num_layer: ", num_layer)
        print("INIT - decoder - dec_units: ", dec_units)
        print("INIT - decoder - cell_type: ", cell_type)
        print("INIT - decoder - dropout: ", dropout)
        print("INIT - decoder - enc_type: ", enc_type)
        print("INIT - decoder - attn_type: ", attn_type)
        print("INIT - decoder - beam_width: ", beam_width)
        print("INIT - decoder - length_penalty_weight: ", length_penalty_weight)
        print("INIT - decoder - pre_embedding: ", pre_embedding)
        print("INIT - decoder - vocab_size: ", vocab_size)
        print("INIT - decoder - embedding_dim: ", embedding_dim)
        print("INIT - decoder - embedding_trainable: ", embedding_trainable)
        print("INIT - decoder - batch_sz: ", batch_sz)
        print("INIT - decoder - max_length_input: ", max_length_input)
        print("INIT - decoder - max_length_output: ", max_length_output)


        if self.pre_embedding == None:
            self.embedding  = tf.keras.layers.Embedding(self.vocab_size,
                                                        self.embedding_dim,
                                                        trainable=True)
        else:
            embedding_matrix = np.load(self.pre_embedding)
            init = tf.keras.initializers.Constant(embedding_matrix)
            self.embedding  = tf.keras.layers.Embedding(self.vocab_size,
                                                        self.embedding_dim,
                                                        embeddings_initializer=init,
                                                        trainable=self.embedding_trainable)


        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(self.vocab_size)

         # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = _create_cell(self.dec_units, self.cell_type, self.num_layer, dropout=self.dropout)

        # Sampler
        self.train_sampler = tfa.seq2seq.sampler.TrainingSampler()
        # else:  # EVAL & TEST
        #   self.sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = build_attention_mechanism(self.dec_units,
            None, self.batch_sz*[self.max_length_input], self.attn_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                                     self.attention_mechanism, attention_layer_size=self.dec_units)

        # Define the decoder with respect to fundamental rnn cell
        self.train_decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.train_sampler, output_layer=self.fc)

    def call(self, dec_input, initial_state, start_token=1, end_token=2, training=False):
        # batch_size should not be specified
        # if fixed, then the redundant evaluation data will make error
        # it may related to bug of tensorflow api
        print("TRAINING - Decoder: ", training)
        if training:
            decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=self.batch_sz, dtype=tf.float32)
            print("decoder_initial_state: ", decoder_initial_state)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=initial_state)

            print("training - decoder_initial_state: ", decoder_initial_state)
            embd_input = self.embedding(dec_input)
            print("training - embd_input.shape: ", embd_input.shape)
            print("training - max_length_output: ", self.max_length_output)
            outputs, _, _ = self.train_decoder(embd_input, initial_state=decoder_initial_state,
                                               sequence_length=self.batch_sz*[self.max_length_output-1])

            print("training - outputs.shape: ", outputs.rnn_output.shape)
        else:
            #decoder_initial_state = self.build_initial_state(self.beam_width*self.batch_sz,
            #    initial_state, tf.float32)
            start_tokens = tf.fill([self.batch_sz], start_token)
            print("start_tokens.shape: ", start_tokens.shape)
            print("end_token: ", end_token)
            print("initial_state: ", initial_state)
            print("dec_input: ", dec_input)

            # set decoder_inital_state which is an AttentionWrapperState considering beam_width
            hidden_state = tfa.seq2seq.tile_batch(
                initial_state, multiplier=self.beam_width)
            decoder_initial_state = self.rnn_cell.get_initial_state(
                batch_size=self.beam_width*self.batch_sz, dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=hidden_state)
            print("decoder_initial_state.cell_state: ", decoder_initial_state.cell_state.shape)
            print("decoder_initial_state.attention: ", decoder_initial_state.attention.shape)
            print("decoder_initial_state.alignments: ", decoder_initial_state.alignments.shape)
            print("decoder_initial_state.attention_state: ", decoder_initial_state.attention_state.shape)

            # Instantiate BeamSearchDecoder
            decoder_instance = tfa.seq2seq.BeamSearchDecoder(
                self.rnn_cell, beam_width=self.beam_width, output_layer=self.fc, length_penalty_weight=self.length_penalty_weight, maximum_iterations=10)
            decoder_embedding_matrix = self.embedding .variables[0]
            print("decoder_embedding_matrix: ", decoder_embedding_matrix.shape)

            # The BeamSearchDecoder object's call() function takes care of everything.
            outputs, final_state, sequence_lengths = decoder_instance(
                decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
            # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
            # The final beam predictions are stored in outputs.predicted_id
            # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
            # final_state = tfa.seq2seq.BeamSearchDecoderState object.
            # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
            print(
                "sequence_lengths.shape = [inference_batch_size, beam_width]: ", sequence_lengths.shape)
            # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
            # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
            # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
            print("outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width): ",
                  outputs.predicted_ids.shape)
            print("outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width): ",
                  outputs.beam_search_decoder_output.scores.shape)
            print(type(outputs.beam_search_decoder_output))
            print("final_state", final_state)

            final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
            beam_scores = tf.transpose(
                outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))
            print("final_outputs.shape = (inference_batch_size, beam_width, time_step_outputs) ",
                  final_outputs.shape)

            outputs = final_outputs[:, 0, :]  # [batch, length]
            print("outputs.shape = [batch, length]: ", outputs.shape)
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

    

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype)
        print("decoder_initial_state: ", decoder_initial_state)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)
        return decoder_initial_state
