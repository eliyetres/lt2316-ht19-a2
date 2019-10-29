import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils import data
import config
from rnn import RNN
from dataset import Dataset
import joblib

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device(config.DEVICE)
#cudnn.benchmark = True

# Parameters
params = {'batch_size': 200,
          'shuffle': True}
max_epochs = config.RNN_NUM_EPOCHS

# Datasets
train_data = joblib.load('train_data.pkl')

#Split into sentences and boundaries
first_sentences_raw = []
second_sentences_raw = []
first_sentences = []
second_sentences = []
boundaries = []

for k in train_data.keys():
    first_sentences_raw.append(train_data[k]['sent1'])
    second_sentences_raw.append(train_data[k]['sent2'])

    first_sentences.append(train_data[k]['sent1_embedding'])
    second_sentences.append(train_data[k]['sent2_embedding'])
    boundaries.append(train_data[k]['boundary'])

print("First sents: %s" % len(first_sentences_raw))
print("Second sents: %s" % len(second_sentences_raw))
print("First sents embedded: %s" % len(first_sentences))
print("Second sents embedded: %s" % len(second_sentences))
print("Boundaries: %s" % len(boundaries))

    
# Generators
training_set1 = Dataset(first_sentences, boundaries)
training_generator1 = data.DataLoader(training_set1, **params)

training_set2 = Dataset(second_sentences, boundaries)
training_generator2 = data.DataLoader(training_set2, **params)

#One vocab for all sentences (should perhaps be in another file)
def get_vocab(sentence_set1, sentence_set2):

    vocabulary = {}
    
    for sent in sentence_set1:
        for word in sent:
            if word in vocabulary:
                vocabulary[word] +=1
            else:
                vocabulary[word] = 1

    for sent in sentence_set2:
        for word in sent:
            if word in vocabulary:
                vocabulary[word] +=1
            else:
                vocabulary[word] = 1
                
    return vocabulary

# Initializing parameters for model
hidden_size = config.RNN_HIDDEN_SIZE #should be an argument
output_size = 2 #same or change
vocab_size = len(get_vocab(first_sentences_raw, second_sentences_raw)) + 1

# Initializing model

#For first sentence in a pair
model1 = RNN(len(first_sentences), hidden_size, output_size, vocab_size)
model1.setup()
model1 = model1.to(device)

#For second sentence in a pair
model2 = RNN(len(second_sentences), hidden_size, output_size, vocab_size)
model2.setup()
model2 = model2.to(device)

# Loop over epochs
for epoch in range(max_epochs):
    print("Epoch %s" % epoch)
    # Training
    for local_batch, local_labels in training_generator1:
        # Transfer to GPU
        print("Sends batch of gen1 to device...")
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        print("Trains model 1...")
        model1.train(local_batch, local_labels)

    for local_batch, local_labels in training_generator2:
        # Transfer to GPU
        print("Sends batch of gen2 to device...")
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        print("Trains model 2...")
        model2.train(local_batch, local_labels)
