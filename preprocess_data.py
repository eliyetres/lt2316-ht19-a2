import csv
import torch
import joblib
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

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


def split_data(sentences, size=0.2):

    test_dict = {}
    train_dict = {}

    # Sentences is a dictionary, keys are integers
    sentence_keys = list(sentences.keys())
    print("Total length of sentences: ".format(len(sentences)))

    # Splitting data into sets
    print("Splitting data into training and tests sets...")
    trainset, testset = train_test_split(sentence_keys, test_size=size)

    # Printing the lengths of the sets
    print("Length of training set: {}".format(len(trainset)))
    print("Length of test set: {}".format(len(testset)))

    # Putting the corresponding values into the new sets as dictionaries
    for k in trainset:
        train_dict[k] = sentences[k]

    for j in testset:
        test_dict[j] = sentences[j]

    return train_dict, test_dict


if __name__ == '__main__':
    vectorized_data = {}
    print("Loading Word2Vec model...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Getting sentence embeddings...")
    # with open(config.CSV_FILENAME, 'r') as rfile:
    with open(config.CSV_FILENAME, 'r') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            try:
                if index == 0:
                    print("index = 0")
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

    # Default test size
    test_size = 0.2
    print("Splitting data into training and testing sets, {}/{}.".format(
        round(100 - (test_size * 100)), round(test_size * 100)))
    trainset, testset = split_data(vectorized_data, test_size)

    joblib.dump(trainset, 'train_data_full.pkl')
    joblib.dump(testset, 'test_data_full.pkl')
