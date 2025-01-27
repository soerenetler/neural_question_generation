import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return dict(

        # File path
        # None or path to pretrained embedding
        pre_embedding=None,#'data/processed/glove_embedding.npy',

        # NN params
        voca_size=34004,
        embedding_size=300,
        embedding_trainable=True,
        hidden_size=512,
        cell_type='gru',  # 'lstm' or 'gru'
        enc_type='mono',  # 'bi' or 'mono'
        num_layer=1,
        dropout=0.4,
        attn='luong',  # 'bahdanau', 'normed_bahdanau', 'luong', 'scaled_luong'
        beam_width=3,
        length_penalty_weight=2.1,

        # Extra params
        ##dtype=tf.float32,
        maxlen_s=60,
        maxlen_dec_train=32,
        maxlen_dec_dev=32,
        start_token=1,  # <GO> index
        end_token=2,  # <EOS> index

        # Learning params
        batch_size=64,
        ##learning_rate=0.001,
        ##decay_step=None,
        ##decay_rate=0.5,
        ##sample_prob=0.25,
    )


def other_params():
    hparams = basic_params()
    hparams.voca_size = 30004
    hparams.embedding = None
    hparams.embedding_trainable = True
    hparams.hidden_size = 300
    hparams.encoder_layer = 1
    hparams.decoder_layer = 1

    hparams.rnn_dropout = 0.3

    hparams.batch_size = 128

    hparams['decay'] = 0.4  # learning rate decay factor
    return hparams
