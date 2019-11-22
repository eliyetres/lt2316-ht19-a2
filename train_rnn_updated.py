import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.optim import Adam
from torch.utils import data

import config
from attention_rnn import SpeakerAttentionClassifier, SpeakerAttentionRNN
from dataset_updated import Dataset
from rnn import SpeakerClassifier, SpeakerRNN
from utils import (generate_batch_vectors, generate_sent_vector,
                   get_boundary_mapping, get_word_vector, load_model,
                   read_data_from_csv)

use_attention = config.USE_ATTENTION


if __name__ == '__main__':
    print("Loading pre-trained embeddings...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Loading training data...")
    if config.RNN_EQUALIZE_CLASS_COUNTS is True:
        print("\tEqualizing class counts!")

    train_data = read_data_from_csv(
        filename=config.CSV_FILENAME_TRAIN,
        train=True,
        equalize=config.EQUALIZE_CLASS_COUNTS,
        num_records=config.RNN_NUM_RECORDS
    )

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

    if use_attention is True:
        print("\tUsing attention!")
        model1 = SpeakerAttentionRNN(
            emb_size=300,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=1,
            dev=device
        )
        model2 = SpeakerAttentionRNN(
            emb_size=300,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=1,
            dev=device
        )
        classifier = SpeakerAttentionClassifier(
            hidden_size=config.RNN_HIDDEN_SIZE * 2,
            num_classes=1
        )
        classifier.to(device)

    else:
        model1 = SpeakerRNN(
            device=device,
            emb_size=300,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_classes=1,
            batch_size=config.RNN_BATCH_SIZE,
            num_layers=1,
            bidirectionality=False
        )
        model2 = SpeakerRNN(
            device=device,
            emb_size=300,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_classes=1,
            batch_size=config.RNN_BATCH_SIZE,
            num_layers=1,
            bidirectionality=False
        )
        classifier = SpeakerClassifier(
            device=device,
            input_size=config.RNN_HIDDEN_SIZE * 2,
            output_size=1
        )
    model1 = model1.to(device)
    optimizer1 = Adam(model1.parameters(), lr=config.RNN_LEARNING_RATE)

    model2 = model2.to(device)
    optimizer2 = Adam(model2.parameters(), lr=config.RNN_LEARNING_RATE)

    classifier = classifier.to(device)
    criterion = nn.BCELoss()

    print("Training model...")
    for epoch in range(config.RNN_NUM_EPOCHS):
        print("Current epoch: {}".format(epoch + 1))
        epoch_loss = 0.0
        for sent1_batch, sent2_batch, boundary_batch in train_generator:
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if use_attention is True:
                # when using attention, seq_len for both sentences need to be the same
                max_sent_len = max(
                    max([len(word_tokenize(a)) for a in sent1_batch]),
                    max([len(word_tokenize(a)) for a in sent2_batch])
                )
                sent1_batch_vectors = generate_batch_vectors(sent1_batch, w2v_model, max_sent_len=max_sent_len)
                sent2_batch_vectors = generate_batch_vectors(sent2_batch, w2v_model, max_sent_len=max_sent_len)
            else:
                sent1_batch_vectors = generate_batch_vectors(sent1_batch, w2v_model)
                sent2_batch_vectors = generate_batch_vectors(sent2_batch, w2v_model)
            boundary_batch = get_boundary_mapping(boundary_batch)

            sent1_batch_vectors = sent1_batch_vectors.to(device)
            sent2_batch_vectors = sent2_batch_vectors.to(device)
            boundary_batch = torch.Tensor(boundary_batch).to(device)

            output1, hidden1 = model1(sent1_batch_vectors)
            output2, hidden2 = model2(sent2_batch_vectors)

            if use_attention is True:
                output = classifier(output1, output2)
            else:
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

    if use_attention is True and config.RNN_EQUALIZE_CLASS_COUNTS is True:
        torch.save(model1, config.RNN_EQ_ATTENTION_MODEL1)
        torch.save(model2, config.RNN_EQ_ATTENTION_MODEL2)
        torch.save(classifier, config.RNN_EQ_ATTENTION_CLASSIFIER)
    elif use_attention is True and config.RNN_EQUALIZE_CLASS_COUNTS is False:
        torch.save(model1, config.RNN_ATTENTION_MODEL1)
        torch.save(model2, config.RNN_ATTENTION_MODEL2)
        torch.save(classifier, config.RNN_ATTENTION_CLASSIFIER)
    elif use_attention is False and config.RNN_EQUALIZE_CLASS_COUNTS is True:
        torch.save(model1, config.RNN_EQ_MODEL1)
        torch.save(model2, config.RNN_EQ_MODEL2)
        torch.save(classifier, config.RNN_EQ_CLASSIFIER)
    else:
        torch.save(model1, config.RNN_MODEL1)
        torch.save(model2, config.RNN_MODEL2)
        torch.save(classifier, config.RNN_CLASSIFIER)
