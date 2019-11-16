import config
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import csv
import pickle
import torch
from sklearn.metrics import classification_report


def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def read_data_from_csv(filename):
    training_data = []
    with open(filename, 'r', encoding='utf8') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            training_data.append({
                'sent1': row[0].strip(),
                'sent2': row[1].strip(),
                'boundary': row[2]
            })
            if index > config.RNN_NUM_RECORDS:
                break

    return training_data


def print_evaluation_score(y_true, y_pred):
    # accuracy, precision, recall, and F-measure.
    target_names = ['Same', 'Change']
    print(classification_report(y_true, y_pred, target_names=target_names))


def get_word_vector(token, model):
    # this is for the padding vector
    if token == '0':
        return torch.zeros(300)
    try:
        return torch.Tensor(model[token])
    except KeyError:
        try:
            return torch.Tensor(model[token.lower()])
        except KeyError:
            return torch.Tensor(config.UNKNOWN_WORD_VECTOR)


def generate_sent_vector(sent, model):
    sent_vector = []
    for token in sent:
        word_vector = get_word_vector(token, model)
        sent_vector.append(word_vector)

    return torch.stack(sent_vector)


def generate_batch_vectors(sent_batch, model):
    sent_vectors = []
    sent_batch_tokenized = [word_tokenize(s) for s in sent_batch]
    max_sent_len = len(max(sent_batch_tokenized, key=lambda x: len(x)))
    for sent in sent_batch_tokenized:
        # padding the sentences with 0
        if len(sent) < max_sent_len:
            for i in range(len(sent), max_sent_len):
                sent.append('0')
        sent_vector = generate_sent_vector(sent, model)
        sent_vectors.append(sent_vector)

    return torch.stack(sent_vectors)


def get_boundary_mapping(boundary_batch):
    mapped_boundaries = [config.BOUNDARY_TO_INT_MAPPING[b]
                         for b in boundary_batch]
    return mapped_boundaries
