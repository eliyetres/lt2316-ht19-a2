import torch
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertTokenizer
from tqdm import trange
import tarfile

import config
from utils import read_data_from_csv, prepare_data_bert


if __name__ == '__main__':
    # train_data is the same thing as the train_data and test_data outputs from preprocess_data, just pickled
    # This helps avoid having to run the preprocess_data script everytime
    print("Loading data...")
    if config.EQUALIZE_CLASS_COUNTS is True:
        print("Equalizing class counts!")
    train_data = read_data_from_csv(
        filename=config.CSV_FILENAME_TRAIN,
        train=True,
        num_records=config.BERT_NUM_RECORDS,
        equalize=config.EQUALIZE_CLASS_COUNTS
    )

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
    train_dataloader = prepare_data_bert(train_data, tokenizer, max_sent_len)

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
