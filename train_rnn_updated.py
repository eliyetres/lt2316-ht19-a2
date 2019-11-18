import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data

import config
from dataset_updated import Dataset
from rnn import SpeakerClassifier, SpeakerRNN
from utils import (generate_batch_vectors, generate_sent_vector,
                   get_boundary_mapping, get_word_vector, load_model, read_data_from_csv)

if __name__ == '__main__':
    print("Loading pre-trained embeddings...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Loading training data...")
    train_data = read_data_from_csv(config.CSV_FILENAME_TRAIN, train=True)

    print("\tTotal length of training data: {}".format(len(train_data)))
    print("\tNumber of SAME records: {}".format(len([a for a in train_data if a['boundary'] == '[SAME]'])))
    print("\tNumber of CHANGE records: {}".format(len([a for a in train_data if a['boundary'] == '[CHANGE]'])))

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
    optimizer1 = Adam(model1.parameters(), lr=config.RNN_LEARNING_RATE)

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
    optimizer2 = Adam(model2.parameters(), lr=config.RNN_LEARNING_RATE)

    classifier = SpeakerClassifier(
        device=device,
        input_size=config.RNN_HIDDEN_SIZE * 2,
        output_size=1
    )
    classifier = classifier.to(device)
    criterion = nn.BCELoss()

    print("Training model...")
    for epoch in range(config.RNN_NUM_EPOCHS):
        print("Current epoch: {}".format(epoch + 1))
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

    torch.save(model1, 'model_1.pkl')
    torch.save(model2, 'model_2.pkl')
    torch.save(classifier, 'classifier.pkl')
