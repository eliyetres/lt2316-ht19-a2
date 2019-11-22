import csv
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import trange
import numpy as np
import tarfile

import config


def read_data_from_csv(filename):
    data = {}
    with open(filename, 'r') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            try:
                # stop at 50,000 records
                if index > config.BERT_NUM_RECORDS:
                    break

                if index % 10000 == 0 and index > 0:
                    print("{} rows processed!".format(index))
                if index == 0:
                    continue
                if index == 1:
                    prev_sent = row[0]
                    boundary = row[1]
                    continue
                next_sent = row[0]

                data[index] = {}
                data[index]['sent1'] = prev_sent
                data[index]['sent2'] = next_sent
                data[index]['boundary'] = config.BOUNDARY_TO_INT_MAPPING[
                    boundary.strip()]

                # set prev_sent to current sentence. And get boundary from this record too
                prev_sent = next_sent
                boundary = row[1]
            except IndexError:
                break

    return data


def combine_sents_bert_style(sent1, sent2):
    sent = "[CLS] " + sent1 + " [SEP] " + sent2 + " [SEP]"
    return sent


def preprocess_sents_bert_style(sent_data, tokenizer, max_sent_len):
    sents = []
    labels = []
    for key in sent_data.keys():
        sent1 = sent_data[key]['sent1']
        sent2 = sent_data[key]['sent2']
        combined_sent = combine_sents_bert_style(sent1, sent2)
        tokenized_sent = tokenizer.tokenize(combined_sent)
        # clip tokens to max_sent_len
        tokenized_sent = tokenized_sent[:max_sent_len]
        sents.append(tokenized_sent)
        labels.append(sent_data[key]['boundary'])
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


def prepare_data(data, tokenizer, max_sent_len):
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


def flat_accuracy(preds, labels):
    # pred_flat = np.argmax(preds, axis=1).flatten()
    # labels_flat = labels.flatten()
    # f1_value = f1_score(labels_flat, pred_flat)
    accuracy = np.sum(preds == labels) / len(labels)
    return accuracy


if __name__ == '__main__':
    # train_data is the same thing as the train_data and test_data outputs from preprocess_data, just pickled
    # This helps avoid having to run the preprocess_data script everytime
    print("Loading data...")
    data = read_data_from_csv(config.CSV_FILENAME)

    # do train-test split
    train_keys, test_keys = train_test_split(list(data.keys()), test_size=0.2, shuffle=True)

    # create train and test dicts
    train_data, test_data = {}, {}
    for key in train_keys:
        train_data[key] = data[key]

    for key in test_keys:
        test_data[key] = data[key]

    del data

    print("Loading models...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # initialize the model with 2 output classes
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device(config.DEVICE)
    model = model.to(device)
    # initialize the optimzier
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_params, lr=2e-5, warmup=.1)

    # INPUTS
    print("Preparing training data...")
    max_sent_len = config.BERT_MAX_SENT_LEN
    train_dataloader = prepare_data(train_data, tokenizer, max_sent_len)

    print("Preparing testing data...")
    test_dataloader = prepare_data(test_data, tokenizer, max_sent_len)

    # TRAINING
    print("Training the model...")
    train_loss_set = []
    for _ in trange(config.BERT_NUM_EPOCHS, desc='Epoch'):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # move batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # unpack batch
            input_ids, segment_masks, attention_masks, labels = batch
            # set optimizer to zero
            optimizer.zero_grad()
            # forward pass
            loss = model(input_ids.long(), token_type_ids=segment_masks,
                         attention_mask=attention_masks, labels=labels)
            # add loss
            train_loss_set.append(loss.item())
            # compute gradients
            loss.backward()
            # backprop
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

    print("Saving model and tokenizer...")
    model_to_save = model.module if hasattr(model, 'module') else model
    # apparently, we HAVE to keep this particular filenames - although we can change the paths I think
    torch.save(model_to_save.state_dict(), config.BERT_MODEL_FILE)
    model_to_save.config.to_json_file(config.BERT_CONFIG_FILE)
    # package the .bin and .config file into a .tar.gz file
    fp = tarfile.open(config.BERT_TAR_FILE, "w:gz")
    fp.add(config.BERT_MODEL_FILE)
    fp.add(config.BERT_CONFIG_FILE)
    fp.close()
    # save the vocabulary - this will create a file called "vocab.txt" at given path
    tokenizer.save_vocabulary('.')

    del model
    del tokenizer

    print("Loading pre-trained model and tokenizer...")
    model = BertForSequenceClassification.from_pretrained(config.BERT_TAR_FILE, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(config.BERT_VOCAB_FILE, do_lower_case=True)

    print("Evaluating the model...")
    model.eval()
    model.to(device)
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    actual_labels = []
    predicted_labels = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_masks, attention_masks, labels = batch
        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_masks,
                           attention_mask=attention_masks)

        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        label_ids = labels.to('cpu').numpy()
        labels_flat = label_ids.flatten()

        # these are used for getting F1 score later
        predicted_labels.extend(pred_flat)
        actual_labels.extend(labels_flat)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Testing data accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print(classification_report(actual_labels, predicted_labels))
