from pickle import dump

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils import data

import config
from dataset import Dataset
from rnn import SpeakerClassifier, SpeakerRNN

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device(config.DEVICE)
#cudnn.benchmark = True

# Parameters
params = {'batch_size': config.RNN_BATCH_SIZE,
          'shuffle': True}
max_epochs = config.RNN_NUM_EPOCHS

# Datasets
print("Loading training data...")
train_data = joblib.load('train_data.pkl')

# Split into sentences and boundaries
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
print("Initializing data generators...")
training_set1 = Dataset(first_sentences, boundaries)
training_generator1 = data.DataLoader(training_set1, **params)

training_set2 = Dataset(second_sentences, boundaries)
training_generator2 = data.DataLoader(training_set2, **params)

# Azfar

print("Initializing models...")
model1 = SpeakerRNN(
    device=device,
    emb_size=first_sentences[0].shape[1],
    hidden_size=config.RNN_HIDDEN_SIZE,
    num_classes=1,
    seq_len=first_sentences[0].shape[0],
    batch_size=config.RNN_BATCH_SIZE,
    num_layers=1,
    bidirectionality=False
)
optimizer1 = Adam(model1.parameters(), lr=0.0001)

model2 = SpeakerRNN(
    device=device,
    emb_size=second_sentences[0].shape[1],
    hidden_size=config.RNN_HIDDEN_SIZE,
    num_classes=1,
    seq_len=second_sentences[0].shape[0],
    batch_size=config.RNN_BATCH_SIZE,
    num_layers=1,
    bidirectionality=False
)
optimizer2 = Adam(model2.parameters(), lr=0.0001)

classifier = SpeakerClassifier(device,config.RNN_HIDDEN_SIZE * 2, 1)
criterion = nn.BCELoss()

# move to GPU
model1.to(device)
model2.to(device)
classifier.to(device)
criterion.to(device)

model1.train()
model2.train()
print("Training the model...")
for epoch in range(config.RNN_NUM_EPOCHS):
    print("Epoch: {}".format(epoch+1))
    epoch_loss = 0.0
    for (batch_seq_1, batch_label_1), (batch_seq_2, batch_label_2) in zip(training_generator1, training_generator2):
        if batch_seq_1.size()[0] != config.RNN_BATCH_SIZE or batch_seq_2.size()[0] != config.RNN_BATCH_SIZE:
            continue   
        # push to GPU
        batch_seq_1,batch_seq_2 = batch_seq_1.to(device), batch_seq_2.to(device)
        batch_label_1, batch_label_2 = batch_label_1.to(device), batch_label_2.to(device)

        output1, hidden1 = model1(batch_seq_1)
        output2, hidden2 = model2(batch_seq_2)
        # push to GPU
        output1,output2=output1.to(device),output2.to(device)
        hidden1,hidden2=hidden1.to(device),hidden2.to(device)

        hidden1 = hidden1.squeeze(dim=0)
        hidden2 = hidden2.squeeze(dim=0)

        combined_output = torch.cat((hidden1, hidden2), dim=1)
        # print(combined_output.size())

        # combined_output = combined_output.view(-1, config.RNN_BATCH_SIZE * config.RNN_HIDDEN_SIZE * 2)
        output = classifier(combined_output)

        # need to reshape this for BCELoss
        batch_label = batch_label_1.view(-1, 1).float()
        batch_label.to(device)
        loss = criterion(output, batch_label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer1.step()
    print("Loss: {}".format(epoch_loss))
    print()

# Save model to disk
with open("trained_model_1.pkl", 'wb+') as tmf1:
    dump(model1, tmf1)
# Save model to disk
with open("trained_model_2.pkl", 'wb+') as tmf2:
    dump(model2, tmf2)
# Save model to disk
with open("classifier.pkl", 'wb+') as tmf3:
    dump(classifier, tmf3)


# Sandra
# One vocab for all sentences (should perhaps be in another file)
# UPDATE: we might not need this
# def get_vocab(sentence_set1, sentence_set2):

#     vocabulary = {}

#     for sent in sentence_set1:
#         for word in sent:
#             if word in vocabulary:
#                 vocabulary[word] += 1
#             else:
#                 vocabulary[word] = 1

#     for sent in sentence_set2:
#         for word in sent:
#             if word in vocabulary:
#                 vocabulary[word] += 1
#             else:
#                 vocabulary[word] = 1

#     return vocabulary


# Initializing parameters for model
# hidden_size = config.RNN_HIDDEN_SIZE  # should be an argument
# output_size = 2  # same or change
# vocab_size = len(get_vocab(first_sentences_raw, second_sentences_raw)) + 1

# # Initializing model

# # For first sentence in a pair
# model1 = RNN(len(first_sentences), hidden_size, output_size, vocab_size)
# model1.setup()
# model1 = model1.to(device)

# # For second sentence in a pair
# model2 = RNN(len(second_sentences), hidden_size, output_size, vocab_size)
# model2.setup()
# model2 = model2.to(device)

# # Loop over epochs
# for epoch in range(max_epochs):
#     print("Epoch %s" % epoch)
#     # Training
#     for local_batch, local_labels in training_generator1:
#         # Transfer to GPU
#         print("Sends batch of gen1 to device...")
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         print("Trains model 1...")
#         model1.train(local_batch, local_labels)

#     for local_batch, local_labels in training_generator2:
#         # Transfer to GPU
#         print("Sends batch of gen2 to device...")
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         print("Trains model 2...")
#         model2.train(local_batch, local_labels)
