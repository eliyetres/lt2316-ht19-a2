import csv
import torch
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

import config


def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def get_word_embedding(token, model):
    try:
        return model[token]
    except KeyError:
        return torch.randn(300)


def get_sent_embedding(sent, model):
    sent_embedding = []
    tokens = word_tokenize(sent)
    for token in tokens:
        token_embedding = get_word_embedding(token, model)
        sent_embedding.append(token_embedding)
    return torch.Tensor(sent_embedding)


if __name__ == '__main__':
    vectorized_data = {}
    print("Loading Word2Vec model...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Getting sentence embeddings...")
    # with open(config.CSV_FILENAME, 'r') as rfile:
    with open("small_parliament_speech_data.csv", 'r') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            try:
                if index == 0:
                    continue
                # if first record, save the sent and boundary, move on to next record
                if index == 1:
                    prev_sent = row[0]
                    boundary = row[1]
                    continue
                next_sent = row[0]

                vectorized_data[index] = {}
                vectorized_data[index]['sent1'] = prev_sent
                vectorized_data[index]['sent1_embedding'] = get_sent_embedding(
                    prev_sent, w2v_model)
                vectorized_data[index]['sent2'] = next_sent
                vectorized_data[index]['sent2_embedding'] = get_sent_embedding(
                    next_sent, w2v_model)
                vectorized_data[index]['boundary'] = config.BOUNDARY_TO_INT_MAPPING[
                    boundary.strip()]

                # set prev_sent to current sentence. And get boundary from this record too
                prev_sent = next_sent
                boundary = row[1]
            except IndexError as e:
                # means we have reached the last row
                break
