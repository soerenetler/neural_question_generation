import nltk
import numpy as np
import tensorflow as tf
import pickle as pkl
import tensorflow_addons as tfa

def remove_eos(sentence, eos='<EOS>', pad='<PAD>'):
    if eos in sentence:
        return sentence[:sentence.index(eos)] + '\n'
    elif pad in sentence:
        return sentence[:sentence.index(pad)] + '\n'
    else:
        return sentence + '\n'


def write_result(predict_results, dic_dir, pred_dir):
    print('Load dic file...')
    with open(dic_dir) as dic:
        dic_file = pkl.load(dic)
    reversed_dic = dict((y, x) for x, y in dic_file.iteritems())

    print('Writing into file...')
    with open(pred_dir, 'w') as f:
        while True:
            try:
                output = predict_results.next()
                output = output['question'].tolist()
                if -1 in output:  # beam search
                    output = output[:output.index(-1)]
                indices = [reversed_dic[index] for index in output]
                sentence = ' '.join(indices)
                sentence = remove_eos(sentence)
                f.write(sentence.encode('utf-8'))

            except StopIteration:
                break

def bleu_score(labels, predictions,
               weights=None, metrics_collections=None,
               updates_collections=None, name=None):

    def _nltk_blue_score(labels, predictions):

        # slice after <eos>
        predictions = predictions.tolist()
        for i in range(len(predictions)):
            prediction = predictions[i]
            if 2 in prediction:  # 2: EOS
                predictions[i] = prediction[:prediction.index(2)+1]

        labels = [
            [[w_id for w_id in label if w_id != 0]]  # 0: PAD
            for label in labels.tolist()]
        predictions = [
            [w_id for w_id in prediction]
            for prediction in predictions]

        return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

    score = tf.py_function(_nltk_blue_score, (labels, predictions), tf.float64)
    return tf.compat.v1.metrics.mean(score * 100)

def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    print("real shape = (BATCH_SIZE, max_length_output): ", real.shape)
    print("pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size ), ", pred.shape)
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    # output 0 for y=0 else output 1
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss

# Build cell for encoder and decoder
def _create_cell(hidden_size, cell_type, num_layer, dropout=0.0):
    rnn_cell = tf.keras.layers.GRUCell(
        hidden_size, dropout=dropout) if cell_type == 'gru' else tf.keras.layers.LSTMCell(hidden_size, dropout=dropout)
    return rnn_cell if num_layer == 1 else tf.keras.layers.StackedRNNCells(([rnn_cell for _ in range(num_layer)]))

def build_attention_mechanism(dec_units, memory, memory_length, attention_type):
    if attention_type == 'bahdanau':
        return tfa.seq2seq.BahdanauAttention(
            units=dec_units,
            memory=memory,
            memory_sequence_length=memory_length)
    elif attention_type == 'normed_bahdanau':
        return tfa.seq2seq.BahdanauAttention(
            units=dec_units,
            memory=memory,
            memory_sequence_length=memory_length,
            normalize=True)

    elif attention_type == 'luong':
        return tfa.seq2seq.LuongAttention(
            units=dec_units,
            memory=memory,
            memory_sequence_length=memory_length)

    elif attention_type == 'scaled_luong':
        return tfa.seq2seq.LuongAttention(
            units=dec_units,
            memory=memory,
            memory_sequence_length=memory_length,
            scale=True)
    else:
        raise ValueError('Unknown attention mechanism : %s' %
                            attention_type)