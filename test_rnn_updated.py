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
    model1 = torch.load("model_1.pkl")
    model2 = torch.load("model_2.pkl")
    classifier = torch.load("classifier.pkl")

    print("Loading pre-trained embeddings...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Loading training data...")
    test_data = read_data_from_csv(config.CSV_FILENAME_TEST, train=False)

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

    print("Evaluating model...")
    for sent1, sent2, boundary in test_generator:

        # If the model was trained using attention
        if use_attention is True:
            # when using attention, seq_len for both sentences need to be the same
            max_sent_len = max(
                    max([len(word_tokenize(a)) for a in sent1]),
                    max([len(word_tokenize(a)) for a in sent2]))
            sent1_vectors = generate_batch_vectors(sent1, w2v_model, max_sent_len=max_sent_len)
            sent2_vectors = generate_batch_vectors(sent2, w2v_model, max_sent_len=max_sent_len)


        # Model without attention
        if use_attention is False:
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

        # #################################################
        # check if it works from here
        #_, predicted_boundary = torch.max(output.data, dim=1)
        _, predicted_boundary = torch.max(output.data, dim=0)

        corr = boundary.item()
        #pred = predicted_boundary[0].item()
        pred = predicted_boundary.item()

        b = "SAME" if corr == 0 else "CHANGE"

        # Predicted sentence
        print(sent1, b, sent2)

        actual.append(corr)
        # if predicted_boundary[0] == boundary:  # correct prediction
        if predicted_boundary.item() == boundary.item():  # correct prediction
            print("Correct")
            predicted.append(corr)
        else:
            print("Incorrect")
            predicted.append(pred)

    print_evaluation_score(actual, predicted)
