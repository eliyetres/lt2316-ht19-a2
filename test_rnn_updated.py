import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from torch.optim import Adam
from torch.utils import data

import config
from dataset_updated import Dataset
from rnn import SpeakerClassifier
from utils import (generate_batch_vectors, generate_sent_vector,
                   get_boundary_mapping, get_word_vector, load_model,
                   print_evaluation_score, read_data_from_csv)

use_attention = config.USE_ATTENTION
# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device(config.DEVICE)

if __name__ == '__main__':
    # Evaluation data
    predicted = []
    actual = []

    # Load trained models and data
    print("Loading models...")
    if use_attention is True and config.RNN_EQUALIZE_CLASS_COUNTS is True:
        print("\tUsing attention!")
        print("\tEqualized class counts!")
        model1 = torch.load(config.RNN_EQ_ATTENTION_MODEL1)
        model2 = torch.save(config.RNN_EQ_ATTENTION_MODEL2)
        classifier = torch.save(config.RNN_EQ_ATTENTION_CLASSIFIER)

    elif use_attention is True and config.RNN_EQUALIZE_CLASS_COUNTS is False:
        print("\tUsing attention!")
        model1 = torch.save(config.RNN_ATTENTION_MODEL1)
        model2 = torch.save(config.RNN_ATTENTION_MODEL2)
        classifier = torch.save(config.RNN_ATTENTION_CLASSIFIER)

    elif use_attention is False and config.RNN_EQUALIZE_CLASS_COUNTS is True:
        print("\tEqualized class counts!")
        model1 = torch.save(config.RNN_EQ_MODEL1)
        model2 = torch.save(config.RNN_EQ_MODEL2)
        classifier = torch.save(config.RNN_EQ_CLASSIFIER)

    else:
        model1 = torch.save(config.RNN_MODEL1)
        model2 = torch.save(config.RNN_MODEL2)
        classifier = torch.save(config.RNN_CLASSIFIER)

    print("Loading pre-trained embeddings...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Loading training data...")
    test_data = read_data_from_csv(config.CSV_FILENAME_TEST, train=False, equalize=False)

    print("\tTotal length of test data: {}".format(len(test_data)))
    print("\tNumber of SAME records: {}".format(len([a for a in test_data if a['boundary'] == '[SAME]'])))
    print("\tNumber of CHANGE records: {}".format(len([a for a in test_data if a['boundary'] == '[CHANGE]'])))

    print("Creating data generator...")
    test_set = Dataset(test_data)
    test_generator = data.DataLoader(
        dataset=test_set,
        drop_last=True,
        batch_size=1,
        shuffle=False)

    # GPU
    classifier = classifier.to(device)
    model1.to(device)
    model2.to(device)

    # Set parameters to eval mode
    model1.eval()
    model2.eval()
    classifier.eval()

    print("Evaluating model...")
    index = 0
    for sent1, sent2, boundary in test_generator:

        if index % 30000 == 0 and index > 0:
            print("\t{}/{} records processed!".format(index, len(test_data)))

        # If the model was trained using attention
        if use_attention is True:
            # when using attention, seq_len for both sentences need to be the same
            max_sent_len = max(
                max([len(word_tokenize(a)) for a in sent1]),
                max([len(word_tokenize(a)) for a in sent2]))
            sent1_vectors = generate_batch_vectors(sent1, w2v_model, max_sent_len=max_sent_len)
            sent2_vectors = generate_batch_vectors(sent2, w2v_model, max_sent_len=max_sent_len)

        # Model without attention
        else:
            sent1_vectors = generate_batch_vectors(sent1, w2v_model)
            sent2_vectors = generate_batch_vectors(sent2, w2v_model)

        boundary = get_boundary_mapping(boundary)
        # to GPU
        sent1_vectors = sent1_vectors.to(device)
        sent2_vectors = sent2_vectors.to(device)
        boundary = torch.Tensor(boundary).to(device)
        # forward
        output1, hidden1 = model1(sent1_vectors)
        output2, hidden2 = model2(sent2_vectors)
        # attention does not combine output
        if use_attention is True:
            output = classifier(output1, output2)

        if use_attention is False:
            hidden1 = hidden1.squeeze(dim=0)
            hidden2 = hidden2.squeeze(dim=0)
            combined_hidden = torch.cat([hidden1, hidden2], dim=1)
            output = classifier(combined_hidden)

        output = output.squeeze(dim=1)
        pred = int(output.item() >= 0.5)
        y_true = boundary.item()

        predicted.append(pred)
        actual.append(y_true)

        index += 1

    print_evaluation_score(actual, predicted)
