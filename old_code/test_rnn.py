import pickle

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.optim import Adam
from torch.utils import data

import config
from dataset import Dataset
from rnn import SpeakerClassifier

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device(config.DEVICE)
#cudnn.benchmark = True

def load_pickle(filename):
    pickle_load = pickle.load(open(filename, 'rb'))
    return pickle_load

# Parameters
params = {'batch_size': 1,
          'shuffle': False}

# Datasets
print("Loading training data...")
test_data = joblib.load('test_data.pkl')

# Split into sentences and boundaries
first_sentences_raw = []
second_sentences_raw = []
first_sentences = []
second_sentences = []
boundaries = []

for k in test_data.keys():
    first_sentences_raw.append(test_data[k]['sent1'])
    second_sentences_raw.append(test_data[k]['sent2'])

    first_sentences.append(test_data[k]['sent1_embedding'])
    second_sentences.append(test_data[k]['sent2_embedding'])
    boundaries.append(test_data[k]['boundary'])

print("First sents: %s" % len(first_sentences_raw))
print("Second sents: %s" % len(second_sentences_raw))
print("First sents embedded: %s" % len(first_sentences))
print("Second sents embedded: %s" % len(second_sentences))
print("Boundaries: %s" % len(boundaries))

# Generators
print("Initializing data generators...")
training_set1 = Dataset(first_sentences, boundaries)
training_generator1 = data.DataLoader(training_set1, **params)

training_set2 = Dataset(second_sentences, boundaries)
training_generator2 = data.DataLoader(training_set2, **params)
print("Finished generating data.")

print("Loading models...")
model1 = load_pickle("trained_model_1.pkl")
model2 = load_pickle("trained_model_2.pkl")
classifier = load_pickle("classifier.pkl")
model1.eval()
model2.eval()

predicted = []
actual = []

for (batch_seq_1, batch_label_1), (batch_seq_2, batch_label_2) in zip(training_generator1, training_generator2):
    # push to GPU
    batch_seq_1 = batch_seq_1.to(device)
    batch_label_1 = batch_label_1.to(device)
    batch_seq_2 = batch_seq_2.to(device)
    batch_label_2 = batch_label_2.to(device)

    output1, hidden1 = model1(batch_seq_1)
    output2, hidden2 = model2(batch_seq_2)
    # push to GPU
    output1, output2 = output1.to(device), output2.to(device)
    hidden1, hidden2 = hidden1.to(device), hidden2.to(device)

    hidden1 = hidden1.squeeze(dim=0)
    hidden2 = hidden2.squeeze(dim=0)

    combined_output = torch.cat((hidden1, hidden2), dim=1)
    # print(combined_output.size())

    output = classifier(combined_output)

    _, predicted_boundary = torch.max(output.data, dim=1)
    # batch_label = batch_label_1.view(-1, 1).float()

    actual.append(batch_label_1.item())
    if predicted_boundary[0] == batch_label_1:  # correct prediction
        # print("Correct")
        predicted.append(batch_label_1.item())

    else:
        # print("Incorrect")
        predicted.append(predicted_boundary[0].item())


def print_evaluation_score(y_true, y_pred):
    # accuracy, precision, recall, and F-measure.
    target_names = ['Same', 'Change']
    print(classification_report(y_true, y_pred, target_names=target_names))


print_evaluation_score(actual, predicted)
