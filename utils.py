import csv
import torch
from random import shuffle
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import classification_report

import config


def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def read_data_from_csv(filename, equalize=False, train=True, num_records=-1):
    data = []
    # only need this stuff for generating class-equalized data for training
    if equalize is True:
        num_change = int(num_records / 2)
        num_same = num_change + config.RNN_SAME_ADDITIONAL_RECORDS
        change_records = 0
        same_records = 0
    with open(filename, 'r', encoding='utf8') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            # if equalize = True, data should be equalized for SAME and CHANGE
            if equalize is True:
                current_boundary = row[2].strip()

                if (current_boundary == '[SAME]' and same_records < num_same) or \
                        (current_boundary == '[CHANGE]' and change_records < num_change):
                    data.append({
                        'sent1': row[0].strip(),
                        'sent2': row[1].strip(),
                        'boundary': current_boundary
                    })

                if current_boundary == '[SAME]':
                    same_records += 1
                elif current_boundary == '[CHANGE]':
                    change_records += 1

                if same_records >= num_same and change_records >= num_change:
                    break
            # just add data as is. This will happen when equalize=False
            else:
                data.append({
                    'sent1': row[0].strip(),
                    'sent2': row[1].strip(),
                    'boundary': row[2].strip()
                })

            # if generating non-equalized training data, stopping condition should be num_records
            if train is True and equalize is False and index > num_records:
                break

            # if train is False, we don't care about either equalization or a max number of records

    shuffle(data)
    return data


def print_evaluation_score(y_true, y_pred):
    # accuracy, precision, recall, and F-measure.
    target_names = ['Same', 'Change']
    print(classification_report(y_true, y_pred, target_names=target_names))


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


def generate_batch_vectors(sent_batch, model, max_sent_len=None):
    sent_vectors = []
    sent_batch_tokenized = [word_tokenize(s) for s in sent_batch]
    if max_sent_len is None:
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
    mapped_boundaries = [config.BOUNDARY_TO_INT_MAPPING[b]
                         for b in boundary_batch]
    return mapped_boundaries


def combine_sents_bert_style(sent1, sent2):
    sent = "[CLS] " + sent1 + " [SEP] " + sent2 + " [SEP]"
    return sent


def preprocess_sents_bert_style(sent_data, tokenizer, max_sent_len):
    sents = []
    labels = []
    for elem in sent_data:
        sent1 = elem['sent1']
        sent2 = elem['sent2']
        combined_sent = combine_sents_bert_style(sent1, sent2)
        tokenized_sent = tokenizer.tokenize(combined_sent)
        # clip tokens to max_sent_len
        tokenized_sent = tokenized_sent[:max_sent_len]
        sents.append(tokenized_sent)
        labels.append(config.BOUNDARY_TO_INT_MAPPING[elem['boundary']])
    return sents, labels


def create_segment_masks(preprocessed_train_data, max_sent_len):
    segment_masks = []
    for sent in preprocessed_train_data:
        sent_id = 0
        sent_mask = []
        for token in sent:
            # append sent_id first, then check for SEP token
            sent_mask.append(sent_id)
            # when SEP token found, switch sent_id to 1, since new sentence is starting now
            if token == '[SEP]':
                sent_id = 1
        while len(sent_mask) < max_sent_len:
            sent_mask.append(0)
        segment_masks.append(sent_mask)
    return segment_masks


def prepare_data_bert(data, tokenizer, max_sent_len):
    print("\tTokenizing data...")
    preprocessed_data, labels = preprocess_sents_bert_style(
        data, tokenizer, max_sent_len)

    # max_sent_len = max(len(a) for a in preprocessed_data)
    # max sentence length for BERT is 512

    print("\tGetting numeric representations, padding and creating segment and attention masks...")
    # get numeric representations of tokens
    input_ids = [tokenizer.convert_tokens_to_ids(
        x) for x in preprocessed_data]
    # pad sequences
    padded_seqs = pad_sequence([torch.LongTensor(x)
                                for x in input_ids], batch_first=True)
    # create segment masks for separating two sentences
    segment_masks = create_segment_masks(
        preprocessed_data, padded_seqs.size(1))

    # create attention masks
    attention_masks = []
    for seq in padded_seqs:
        seq_mask = [float(x > 0) for x in seq]
        attention_masks.append(seq_mask)

    print("\tCreating tensors...")
    # make everything a tensor
    tensor_seqs = torch.LongTensor(padded_seqs)
    tensor_labels = torch.LongTensor(labels)
    tensor_attention_masks = torch.LongTensor(attention_masks)
    tensor_segment_masks = torch.LongTensor(segment_masks)

    print("\tCreating dataset...")
    # batching
    batch_size = config.BERT_BATCH_SIZE
    # make an iterator
    tensor_data = TensorDataset(
        tensor_seqs, tensor_segment_masks, tensor_attention_masks, tensor_labels)
    tensor_sampler = RandomSampler(tensor_data)
    tensor_dataloader = DataLoader(
        tensor_data, sampler=tensor_sampler, batch_size=batch_size)

    return tensor_dataloader
