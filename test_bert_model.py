import torch
import numpy as np
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer

import config
from utils import read_data_from_csv, prepare_data_bert, print_evaluation_score


def flat_accuracy(preds, labels):
    accuracy = np.sum(preds == labels) / len(labels)
    return accuracy


if __name__ == '__main__':
    print("Loading data...")

    if config.EQUALIZE_CLASS_COUNTS is True:
        print("Equalized class counts!")
    test_data = read_data_from_csv(
        filename=config.CSV_FILENAME_TEST,
        train=False
    )

    print("Loading models...")
    device = torch.device(config.DEVICE)
    model = BertForSequenceClassification.from_pretrained(config.BERT_TAR_FILE, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(config.BERT_VOCAB_FILE, do_lower_case=True)

    print("Preparing testing data...")
    max_sent_len = config.BERT_MAX_SENT_LEN
    test_dataloader = prepare_data_bert(test_data, tokenizer, max_sent_len)

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

        tmp_eval_accuracy = flat_accuracy(pred_flat, labels_flat)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Testing data accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print_evaluation_score(actual_labels, predicted_labels)
