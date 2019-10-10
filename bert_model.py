import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import joblib

import config


def combine_sents_bert_style(sent1, sent2):
    sent = "[CLS] " + sent1 + " [SEP] " + sent2 + " [SEP]"
    return sent


def preprocess_sents_bert_style(sent_data, tokenizer):
    sents = []
    labels = []
    for key in sent_data.keys():
        sent1 = sent_data[key]['sent1']
        sent2 = sent_data[key]['sent2']
        combined_sent = combine_sents_bert_style(sent1, sent2)
        tokenized_sent = tokenizer.tokenize(combined_sent)
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


if __name__ == '__main__':
    # train_data is the same thing as the train_data and test_data outputs from preprocess_data, just pickled
    # This helps avoid having to run the preprocess_data script everytime
    print("Loading data and models...")
    train_data = joblib.load('train_data.pkl')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    # initialize the model with 2 output classes
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
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
    print("Preparing data...")
    preprocessed_train_data, train_labels = preprocess_sents_bert_style(
        train_data, tokenizer)

    max_sent_len = max(len(a) for a in preprocessed_train_data)
    # get numeric representations of tokens
    input_ids = [tokenizer.convert_tokens_to_ids(
        x) for x in preprocessed_train_data]
    # pad sequences
    padded_seqs = pad_sequence([torch.LongTensor(x)
                                for x in input_ids], batch_first=True)
    # create segment masks for separating two sentences
    train_segment_masks = create_segment_masks(
        preprocessed_train_data, len(padded_seqs[0]))

    # create attention masks
    train_attention_masks = []
    for seq in padded_seqs:
        seq_mask = [float(x > 0) for x in seq]
        train_attention_masks.append(seq_mask)

    # make everything a tensor
    train_seqs = torch.LongTensor(padded_seqs)
    train_labels = torch.LongTensor(train_labels)
    train_attention_masks = torch.LongTensor(train_attention_masks)
    train_segment_masks = torch.LongTensor(train_segment_masks)

    # batching
    batch_size = config.BERT_BATCH_SIZE
    # make an iterator
    train_data = TensorDataset(
        train_seqs, train_segment_masks, train_attention_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)

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