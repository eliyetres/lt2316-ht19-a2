import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

import config
from dataset_updated import Dataset
from rnn import SpeakerRNN, SpeakerClassifier


def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def read_data_from_csv(filename):
    training_data = []
    with open(filename, 'r') as rfile:
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
    mapped_boundaries = [config.BOUNDARY_TO_INT_MAPPING[b] for b in boundary_batch]
    return mapped_boundaries


if __name__ == '__main__':
    print("Loading pre-trained embeddings...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Loading training data...")
    train_data = read_data_from_csv(config.CSV_FILENAME_TRAIN)

    print("Creating data generator...")
    train_set = Dataset(train_data)
    train_generator = data.DataLoader(
        dataset=train_set,
        drop_last=True,
        batch_size=config.RNN_BATCH_SIZE,
        shuffle=True
    )

    print("Initializing models...")
    device = torch.device(config.DEVICE)
    model1 = SpeakerRNN(
        device=device,
        emb_size=300,
        hidden_size=config.RNN_HIDDEN_SIZE,
        num_classes=1,
        batch_size=config.RNN_BATCH_SIZE,
        num_layers=1,
        bidirectionality=False
    )
    model1 = model1.to(device)
    optimizer1 = Adam(model1.parameters(), lr=0.0001)

    model2 = SpeakerRNN(
        device=device,
        emb_size=300,
        hidden_size=config.RNN_HIDDEN_SIZE,
        num_classes=1,
        batch_size=config.RNN_BATCH_SIZE,
        num_layers=1,
        bidirectionality=False
    )
    model2 = model2.to(device)
    optimizer2 = Adam(model2.parameters(), lr=0.0001)

    classifier = SpeakerClassifier(
        device=device,
        input_size=config.RNN_HIDDEN_SIZE * 2,
        output_size=1
    )
    classifier = classifier.to(device)
    criterion = nn.BCELoss()

    print("Training model...")
    for epoch in range(config.RNN_NUM_EPOCHS):
        epoch_loss = 0.0
        for sent1_batch, sent2_batch, boundary_batch in train_generator:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            sent1_batch_vectors = generate_batch_vectors(sent1_batch, w2v_model)
            sent2_batch_vectors = generate_batch_vectors(sent2_batch, w2v_model)
            boundary_batch = get_boundary_mapping(boundary_batch)

            sent1_batch_vectors = sent1_batch_vectors.to(device)
            sent2_batch_vectors = sent2_batch_vectors.to(device)
            boundary_batch = torch.Tensor(boundary_batch).to(device)

            output1, hidden1 = model1(sent1_batch_vectors)
            output2, hidden2 = model2(sent2_batch_vectors)

            hidden1 = hidden1.squeeze(dim=0)
            hidden2 = hidden2.squeeze(dim=0)

            combined_hidden = torch.cat([hidden1, hidden2], dim=1)

            output = classifier(combined_hidden)
            # binary cross entropy needs this; originally the shape of output is [batch_size, 1]
            output = output.squeeze(dim=1)

            loss = criterion(output, boundary_batch)
            epoch_loss += loss.item()
            loss.backward()

            optimizer1.step()
            optimizer2.step()
        print("Loss: {}".format(epoch_loss))
        print()
