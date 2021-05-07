import nltk
import tensorflow as tf


class BleuScore(tf.keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(BleuScore, self).__init__(name=name, **kwargs)
        self.bleu = self.add_weight(name="bleu", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # slice after <eos>
        predictions = y_pred.tolist()
        for i in range(len(predictions)):
            prediction = predictions[i]
            if 2 in prediction:  # 2: EOS
                predictions[i] = prediction[:prediction.index(2)+1]

        labels = [
            [[w_id for w_id in label if w_id != 0]]  # 0: PAD
            for label in y_true.tolist()]
        predictions = [
            [w_id for w_id in prediction]
            for prediction in predictions]

        self.bleu.assign_add(tf.reduce_mean(
            float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))))

    def result(self):
        return self.bleu * 100

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bleu.assign(0.0)
