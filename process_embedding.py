import numpy as np
import pickle as pkl
import os

# Load & process GloVe
data_dir = 'data/'
glove = 'glove.840B.300d'

if not os.path.exists(data_dir + 'processed/' + glove + '.dic.npy'):
    print('Reading original Glove file...')
    with open(data_dir + glove + '.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print('Processing original Glove file to dictionary...\n')
    embedding = dict()
    for line in lines:
        word, coefs = line.split(maxsplit=1)
        embedding[word] = np.fromstring(coefs, "f", sep=" ", count=300)

    # Save Glove as dic file
    np.save(data_dir + 'processed/' + glove + '.dic.npy', embedding)

else:
    print('Glove dictionary exists!')
    print('Loading Glove dictionary...\n')
    embedding = np.load(data_dir + 'processed/' + glove + '.dic.npy', allow_pickle=True).item()

# Make pre-trianed embedding with GloVe
print('Generate pre-trained embedding with Glove')
with open('data/processed/vocab_xinyadu.dic', 'rb') as f:
    vocab = pkl.load(f)

embedding_vocab = np.tile(embedding['UNKNOWN'], [len(vocab), 1])
'''
vocab['<PAD>'] = 0
vocab['<GO>'] = 1
vocab['<EOS>'] = 2
vocab['<UNK>'] = 3
'''
embedding_vocab[0] = 0.0  # vocab['<PAD>'] = 1
embedding_vocab[1] = embedding['<s>']
embedding_vocab[2] = embedding['EOS']
embedding_vocab[3] = embedding['UNKNOWN']

unk_num = 0
for word, idx in vocab.items():
    if word in embedding:
        embedding_vocab[idx] = embedding[word]
    else:
        unk_num += 1


np.save('data/processed/glove_embedding.npy', embedding_vocab)

# check how many unknown words
print('vocab size : %d' % len(embedding_vocab))
print('unknown word size : %d' % unk_num)
