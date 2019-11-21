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
    # this is for the padding vector
    if token == 0:
        return torch.zeros(300)
    try:
        return model[token]
    except KeyError:
        return torch.randn(300)


def get_sent_embedding(sent, max_sent_len, model):
    sent_embedding = []
    tokens = word_tokenize(sent)
    # for token in tokens:
    #     token_embedding = get_word_embedding(token, model)
    #     sent_embedding.append(token_embedding)
    for i in range(0, max_sent_len):
        try:
            token = tokens[i]
            token_embedding = get_word_embedding(token, model)
        except IndexError:
            token_embedding = get_word_embedding(0, model)
        sent_embedding.append(token_embedding)
    return torch.stack([torch.Tensor(a) for a in sent_embedding])


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

    print("Getting sentence pairs from file...")
    # with open(config.CSV_FILENAME, 'r') as rfile:
    rows_to_process = 10000
    max_sent1_len = 0
    max_sent2_len = 0
    with open(config.CSV_FILENAME, 'r') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            try:
                if index == 0:
                    continue

                if index % 1000 == 0:
                    print("\t%d rows processed!" % index)

                if index > rows_to_process:
                    break
                # if first record, save the sent and boundary, move on to next record
                if index == 1:
                    prev_sent = row[0]
                    boundary = row[1]
                    continue
                next_sent = row[0]

                # we use index-2 here so that it is 0-indexed - we skip 0 because it's header row, and we skip 1 because
                # it is first sentence of the dataset
                vectorized_data[index - 2] = {}
                vectorized_data[index - 2]['sent1'] = prev_sent
                # vectorized_data[index]['sent1_embedding'] = get_sent_embedding(
                #     prev_sent, w2v_model)
                vectorized_data[index - 2]['sent2'] = next_sent
                # vectorized_data[index]['sent2_embedding'] = get_sent_embedding(
                #     next_sent, w2v_model)
                vectorized_data[index - 2]['boundary'] = config.BOUNDARY_TO_INT_MAPPING[
                    boundary.strip()]

                sent1_length = len(word_tokenize(vectorized_data[index - 2]['sent1']))
                if sent1_length > max_sent1_len:
                    max_sent1_len = sent1_length

                sent2_length = len(word_tokenize(vectorized_data[index - 2]['sent2']))
                if sent2_length > max_sent2_len:
                    max_sent2_len = sent2_length

                # set prev_sent to current sentence. And get boundary from this record too
                prev_sent = next_sent
                boundary = row[1]
            except IndexError as e:
                # means we have reached the last row
                break

    # get sentence embeddings - we can't do it in the above loop because we need to do the maximum sent length
    # before padding sequences
    print("Generating sentence embeddings...")
    for index in range(len(vectorized_data)):
        sent1 = vectorized_data[index]['sent1']
        sent2 = vectorized_data[index]['sent2']
        vectorized_data[index]['sent1_embedding'] = get_sent_embedding(sent1, max_sent1_len, w2v_model)
        vectorized_data[index]['sent2_embedding'] = get_sent_embedding(sent2, max_sent2_len, w2v_model)

    # Default test size
    test_size = 0.2
    print("Splitting data into training and testing sets, {}/{}.".format(
        round(100 - (test_size * 100)), round(test_size * 100)))
    trainset, testset = split_data(vectorized_data, test_size)

    print("Saving files to disk...")
    joblib.dump(trainset, 'train_data.pkl')
    joblib.dump(testset, 'test_data.pkl')
