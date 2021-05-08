import argparse
import pickle as pkl

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

import model
import params

from utils import remove_eos, write_result, loss_function
from bleu_score import BleuScore

FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'pred'], 'train, eval')
flags.DEFINE_string('train_sentence', '', 'path to the training sentence.')
flags.DEFINE_string('train_question', '', 'path to the training question.')
flags.DEFINE_string('eval_sentence', '', 'path to the evaluation sentence.')
flags.DEFINE_string('eval_question', '', 'path to the evaluation question.')
flags.DEFINE_string('test_sentence', '', 'path to the test sentence.')
flags.DEFINE_string('dic_dir', '', 'path to the dictionary')
flags.DEFINE_string('model_dir', '', 'path to save the model')
flags.DEFINE_string('pred_dir', '', 'path to save the predictions')
flags.DEFINE_string('params', '', 'parameter setting')
flags.DEFINE_integer('num_epochs', 10, 'training epoch size', lower_bound=0)


def main(unused):
    # Load parameters
    model_params = getattr(params, FLAGS.params)()

    # Define estimator
    q_generation = model.QG(model_params)

    q_generation.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function,
                         metrics=[BleuScore()])

    # Training dataset
    train_sentence = np.load(FLAGS.train_sentence)  # train_data
    train_question = np.load(FLAGS.train_question)  # train_label
    TRAIN_BUFFER_SIZE = len(train_sentence)
    train_input_data = tf.data.Dataset.from_tensor_slices((train_sentence, train_question)).shuffle(
        TRAIN_BUFFER_SIZE).batch(model_params['batch_size'], drop_remainder=True)

    # Evaluation dataset
    eval_sentence = np.load(FLAGS.eval_sentence)
    eval_question = np.load(FLAGS.eval_question)
    EVAL_BUFFER_SIZE = len(train_sentence)
    eval_input_data = tf.data.Dataset.from_tensor_slices((eval_sentence, eval_question)).shuffle(
        EVAL_BUFFER_SIZE).batch(model_params['batch_size'], drop_remainder=True)

    # train and evaluate
    if FLAGS.mode == 'train':
        example_input_batch, example_target_batch = next(iter(train_input_data))
        print("Shape train_input_data: ", example_input_batch.shape, example_target_batch.shape)
        q_generation.fit(train_input_data,
                         epochs=FLAGS.num_epochs,)
                         #validation_data=eval_input_data)
        q_generation.summary()

    elif FLAGS.mode == 'eval':
        q_generation.evaluate(eval_input_data)
        # exp_nn.evaluate(delay_secs=0)

    else:  # 'pred'
        # Load test data
        test_sentence = np.load(FLAGS.test_sentence)

        # prediction input function for estimator
        test_input_data = tf.data.Dataset.from_tensor_slices(
            {'enc_inputs': test_sentence}).batch(model_params['batch_size'], drop_remainder=True)

        # prediction
        predict_results = q_generation.predict(test_input_data)

        # write result(question) into file
        write_result(predict_results, FLAGS.dic_dir, FLAGS.pred_dir)


if __name__ == '__main__':
    app.run(main)
