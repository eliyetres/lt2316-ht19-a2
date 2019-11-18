import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils import data

import config
from dataset_updated import Dataset
from rnn import SpeakerClassifier
from utils import (generate_batch_vectors, generate_sent_vector,
                   get_boundary_mapping, get_word_vector, load_model,
                   print_evaluation_score, read_data_from_csv)

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device(config.DEVICE)

if __name__ == '__main__':
    # Evaluation data
    predicted = []
    actual = []

    print("Loading models...")
    model1 = torch.load("model_1.pkl")
    model2 = torch.load("model_2.pkl")
    classifier = torch.load("classifier.pkl")

    # GPU
    classifier = classifier.to(device)
    model1.to(device)
    model2.to(device)

    # Set parameters to eval mode
    model1.eval()
    model2.eval()

    print("Loading pre-trained embeddings...")
    w2v_model = load_model(config.PATH_TO_PRETRAINED_EMBEDDINGS)

    print("Loading training data...")
    train_data = read_data_from_csv(config.CSV_FILENAME_TEST, train=False)

    print("Creating data generator...")
    test_set = Dataset(train_data)
    train_generator = data.DataLoader(
        dataset=test_set,
        drop_last=True,
        batch_size=1,
        shuffle=False)

    print("Evaluating model...")
    for sent1, sent2, boundary in train_generator:
        sent1_vectors = generate_batch_vectors(sent1, w2v_model)
        sent2_vectors = generate_batch_vectors(sent2, w2v_model)
        boundary = get_boundary_mapping(boundary)

        sent1_vectors = sent1_vectors.to(device)
        sent2_vectors = sent2_vectors.to(device)
        boundary = torch.Tensor(boundary).to(device)

        output1, hidden1 = model1(sent1_vectors)
        output2, hidden2 = model2(sent2_vectors)

        hidden1 = hidden1.squeeze(dim=0)
        hidden2 = hidden2.squeeze(dim=0)

        combined_hidden = torch.cat([hidden1, hidden2], dim=1)

        output = classifier(combined_hidden)
        output = output.squeeze(dim=1)

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
