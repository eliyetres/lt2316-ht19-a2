import config
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import csv
import pickle
import torch
from random import shuffle
from sklearn.metrics import classification_report


def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def read_data_from_csv(filename, equalize=False, train=True, num_records=-1):
    data = []
    # only need this stuff for generating class-equalized data for training
    if equalize is True:
        num_change = int(num_records / 2)
        num_same = num_change + config.RNN_SAME_ADDITIONAL_RECORDS
        change_records = 0
        same_records = 0
    with open(filename, 'r', encoding='utf8') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            # if equalize = True, data should be equalized for SAME and CHANGE
            if equalize is True:
                current_boundary = row[2].strip()

                if (current_boundary == '[SAME]' and same_records < num_same) or \
                        (current_boundary == '[CHANGE]' and change_records < num_change):
                    data.append({
                        'sent1': row[0].strip(),
                        'sent2': row[1].strip(),
                        'boundary': current_boundary
                    })

                if current_boundary == '[SAME]':
                    same_records += 1
                elif current_boundary == '[CHANGE]':
                    change_records += 1

                if same_records >= num_same and change_records >= num_change:
                    break
            # just add data as is. This will happen when equalize=False
            else:
                data.append({
                    'sent1': row[0].strip(),
                    'sent2': row[1].strip(),
                    'boundary': row[2].strip()
                })

            # if generating non-equalized training data, stopping condition should be num_records
            if train is True and equalize is False and index > num_records:
                break

            # if train is False, we don't care about either equalization or a max number of records

    shuffle(data)
    return data


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


def generate_batch_vectors(sent_batch, model, max_sent_len=None):
    sent_vectors = []
    sent_batch_tokenized = [word_tokenize(s) for s in sent_batch]
    if max_sent_len is None:
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
