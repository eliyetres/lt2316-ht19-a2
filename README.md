

# LT2316 HT19 Assignment 2

Sentence pair classification for parliamentary turn-taking
From Asad Sayeed's Machine Learning course at the University of Gothenburg.

## Group Broccoli 🥦

Azfar Imtiaz\
Elin Hagman\
Sandra Derbring

## Task Description

The aim of this assignment is to, given a pair of sentences where each sentence is essentially a dialogue, identify whether they were spoken by the same person, or two different persons. So essentially, this can be categorized as a binary classification problem.
The approaches that we selected for this problem are as follows:
- Two unidirectional, one layer RNNs (one for each sentence), followed by a classifier which consists of a single linear layer and a sigmoid
- BERT for sequence classification
- Two unidirectional, one layer RNNs with attention mechanism (one for each sentence), followed by a classifier which consists of a single linear layer and a sigmoid.

## Data processing

### Pre processing

In the pre processing, the paragraphs are first selected by their speech tags and from that the speaker names are extracted. The speeches for the "nospeaker" which is the parliament's President speaking are also included. Then for every sentence, the  boundary SAME was added to all sentences when the speaker does not change. For every new speech, a list is kept of the current and previous speaker to check if they're the same, and popping the list for each new speaker name in the speech tag. If the speaker changes, the boundary is set to `CHANGE`, if not it stays `SAME`. The text is split into training, test and validation files and saved to `.csv`.
The sentences and boundaries were written to a csv file in the format: \
`Sentence1 | Sentence2 | Boundary`

### Post processing

Words like "hon." are changed to their full word equivalent "honorary" to make sure they match their correct word vector in the gensim model. Some British spelled words were changed to their American equivalent, and hyphens were removed for compound words that were spelled with hyphens in British English but not in American. The tokenization had erroneously split some sentences on abbreviations (e.g. "hon.") and this was checked and concatenated. The rest of the words not found in the gensim model at all were removed from the text. Removing some words resulted in some sentences just being an empty string so in the end, these sentences were also removed after making sure that not too much data was lost.

## Training and test data
To create training and testing data, the `read_data` script is used. It loads the speech data from all XML files (the path to which is specified in the config file), and applies the preprocessing techniques upon it as described above. This is followed by applying a train-test split on the entire data, where the train-test ratio is set by the user when running the script (default is 80-20). After this, the training and testing data are both written to separate .csv files (as specified in the config file, as specified under `CSV_FILENAME_TRAIN` and `CSV_FILENAME_TEST`). 

One problem that we noticed during training was that there are a lot more records for `SAME` as opposed to `CHANGE` (which naturally makes sense). This can lead the model to overfitting on the `SAME` class. 
To get around this, we specified a bunch of parameters in the config file:

- The parameter `RNN_NUM_RECORDS` specifies how many records to load from the training file to train the models. This parameter exists because the total amount of records in the training data filename is huge, and using all of it can take a lot of time. 

- The binary parameter `EQUALIZE_CLASS_COUNTS` ensures that if set to True, an equalized amount of class data is loaded from the training file. What this means is that for example, if `RNN_NUM_RECORDS` is 10,000, that means 5000 records for `SAME` and 5000 records for `CHANGE` are loaded from the training data file.

- The parameter `RNN_SAME_ADDITIONAL_RECORDS` allows the user to select an additional number of records to use for the `SAME` class. This is because we have a lot more records for the `SAME` class as opposed to the `CHANGE` class, and if we want that to reflect in the training data while still keeping a reasonable ratio between the number of `SAME` and `CHANGE` records, we can specify how many additional records for the `SAME` class to use.
- So for example, if `EQUALIZE_CLASS_COUNTS` is True, `RNN_NUM_RECORDS` is 10,000 and `RNN_SAME_ADDITIONAL_RECORDS` is 1500, then 5000 records will be loaded for the `CHANGE` class and 6500 records will be loaded for the `SAME` class.

## Approach

### RNN network and classifier models

The recurrent network we designed for this task consists of single GRU layer, which receives as input the sentence embeddings, and returns both the output and hidden states from the GRU. We did not define an embedding layer in this network, as we are using the pre-trained Word2Vec embeddings to generate the word vectors for each word in a given sentence. Therefore, the preprocessing involved before feeding a sentence into the network is as follows: get all sentences in the batch, and get the maximum sentence length from this batch. For each sentence, replace each word with its associated vector as per the pre-trained embeddings, and for all sentences having a shorter word amount than the maximum sentence length for this batch, pad it with the vector associated with the padding vector (which is a 300-dimensional vector consisting of all zeros), and any unknown words will be replaced by the vector associated with unknown words, which is specified in the config file.

Two objects of this same recurrent network class are created, with two optimizers. The criterion used is Binary Cross Entropy, since the possible labels are either 0 or 1 (`SAME` or `CHANGE`).

We have defined a single dataloader which returns both sentence batches simultaneously (where the first batch consists of the first sentence of the inputs, and the second batch consists of the second sentence of the inputs). The first batch is fed into one object, and the second batch into the other object. The hidden states from both these models are combined by concatenating, e.g. if the output is 300-dimensional and there are two outputs, this would result in a 600-dimensional vector, and this is then fed into the classifier model. This final model is used for the actual classification.

Initially two LSTMs were used in the recurrent models, but were changed to GRU since it returns a single hidden state and is easier to deal with. Some structural changes were also made from the initial RNN; it was concluded to not be a good idea to have the optimizer and criterion inside the class, as that would make the weights not update properly.

### RNN network model with Attention

This is the alternate architecture we experimented with, as part of the bonus. The RNN models are almost identical to the ones in the above setting, except that here we take the output states for concatenation from those networks, instead of the hidden states. The classifier model consists of an attention network which takes these concatenated outputs, applies a softmax on them to get probabilities, and returns a weighted output which consists of the computed probabilities multiplied with the output vector. This is then fed to a linear model, followed by a Sigmoid (just like in the above setting).

The idea of adding an attention mechanism to the RNN network is that attention should help the classifier identify which parts of the concatenated output are important for making the prediction. Ideally, that should be words that signify a change or sameness, like "My honour.". If the first sentence includes these words, it's almost always the same speaker (`SAME`).

**NOTE**: The concatenation of the output vectors here is applied on the second dimension, which is basically the output_size dimension. Perhaps an interesting variant would be to apply the concatenation at the word level instead, and see if that has any impact on the results. For future work!

### BERT network model

As the second part of the assignment, we fine-tuned a BERT model on this dataset. For this, we mostly followed this (very helpful!) blog step by step:
[https://mccormickml.com/2019/07/22/BERT-fine-tuning/](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

One thing we did differently was to combine our two input sentences into a single sentence using the CLS and SEP tokens, and used the segment mask to help the model understand where the first sentence ends and the second sentence begins. We have used a 0 for each token of the first sentence, and a 1 for each token of the second sentence.

Otherwise, we pretty much followed the instructions in this blog in a straightforward manner, and that helped us train the model. One thing we faced quite a bit of complication with was saving and loading the BERT model. Since we used an older version of the HuggingFace module (`pytorch_pretrained_bert` instead of `transformers`), it was hard to find instructions for the rather convoluted process of saving the model to disk. The model itself is saved as a .bin file, and its configuration is saved to a separate .json file. These two files are then packaged into a .tar.gz compressed directory, and this is what the model needs to be loaded from.
In the more recent version of the module, the BERT classifier object simply contains a `save_pretrained` function that can be used to save the model to disk (but we discovered this too late).

## Evaluation

### RNN evaluation

The text in the test data is processed in the same way as for the training data. As opposed to multi-class classification, where the class with the highest probability after applying softmax is selected, the prediction for binary classification is computed by getting the probability and assign it to class 1 if it's >=0.5 and assign it to class 0 otherwise. The labels for the correct class and the predicted class are saved and used to generate a classification report showing precision, recall and f1-score using scikit-learn's classification report.

### Results

#### Common configurations for all RNN models

Batch size: 256\
Number of epochs: 10\
Hidden size: 300\
Learning rate: 0.0005

#### RNN trained without attention and without equalized class counts  

Total length of training data: 100000\
Number of SAME records: 87990\
Number of CHANGE records: 12011

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Same         | 0.93      | 0.92   | 0.92     | 507666  |
| Change       | 0.42      | 0.48   | 0.45     | 63852   |
|              |           |        |          |         |
| micro avg    | 0.87      | 0.87   | 0.87     | 571518  |
| macro        | 0.68      | 0.70   | 0.69     | 571518  |
| weighted avg | 0.88      | 0.87   | 0.87     | 571518  |

#### RNN trained without attention and using equalized class counts  

Total length of training data: 110000\
Number of SAME records: 60000\
Number of CHANGE records: 50000

Final loss: 4.4

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Same         | 0.96      | 0.83   | 0.89     | 507666  |
| Change       | 0.34      | 0.69   | 0.46     | 63852   |
|              |           |        |          |         |
| micro avg    | 0.82      | 0.82   | 0.82     | 571518  |
| macro        | 0.65      | 0.76   | 0.67     | 571518  |
| weighted avg | 0.89      | 0.82   | 0.84     | 571518  |

#### RNN trained using attention and without equalized class counts  

Total length of training data: 100000\
Number of SAME records: 87990\
Number of CHANGE records: 12011

Final loss:  4.8

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Same         | 0.92      | 0.99   | 0.95     | 507666  |
| Change       | 0.71      | 0.28   | 0.40     | 63852   |
|              |           |        |          |         |
| micro avg    | 0.91      | 0.91   | 0.91     | 571518  |
| macro        | 0.81      | 0.63   | 0.68     | 571518  |
| weighted avg | 0.89      | 0.91   | 0.89     | 571518  |

#### RNN trained using attention and using equalized class counts

Total length of training data: 110000\
Number of SAME records: 60000\
Number of CHANGE records: 50000

Final loss: 8.58

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Same         | 0.94      | 0.93   | 0.94     | 507666  |
| Change       | 0.50      | 0.54   | 0.52     | 63852   |
|              |           |        |          |         |
| micro avg    | 0.89      | 0.89   | 0.89     | 571518  |
| macro        | 0.72      | 0.74   | 0.73     | 571518  |
| weighted avg | 0.89      | 0.89   | 0.89     | 571518  |

#### BERT model without equalized class counts  

Total length of training data: 100001\
Number of SAME records: 87990\
Number of CHANGE records: 12011

Final loss: 0.091

Testing data accuracy: 0.938

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Same         | 0.95      | 0.98   | 0.97     | 507666  |
| Change       | 0.80      | 0.61   | 0.69     | 63852   |
|              |           |        |          |         |
| micro avg    | 0.94      | 0.94   | 0.94     | 571518  |
| macro        | 0.88      | 0.79   | 0.83     | 571518  |
| weighted avg | 0.93      | 0.94   | 0.94     | 571518  |

#### Common configurations for BERT models

Batch size: 16\
Number of epochs: 4\
Maximum sentence length: 256

#### BERT with equalized class counts

Total length of training data: 110000\
Number of SAME records: 60000\
Number of CHANGE records: 50000

Final loss: 0.052

Testing data accuracy: 0.901

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Same         | 0.98      | 0.91   | 0.94     | 507666  |
| Change       | 0.54      | 0.86   | 0.66     | 63852   |
|              |           |        |          |         |
| micro avg    | 0.90      | 0.90   | 0.90     | 571518  |
| macro avg    | 0.76      | 0.89   | 0.80     | 571518  |
| weighted avg | 0.93      | 0.90   | 0.91     | 571518  |

#### Analysis of results

Because of fear of the model overfitting, the evaluation metrics to focus on here are precision and recall, as opposed to accuracy (since the model can massively overfit on the `SAME` class and can still return a very good accuracy, as the testing set consists of a lot more records of `SAME` than `CHANGE`).

The precision and recall for the `SAME` class are very high in all cases, but it is the `CHANGE` class that we are more interested in. The common pattern is that if we don't use equalized class counts, the precision is quite high for the `CHANGE` class, and the recall is quite low. Using equalized counts brings down the precision, and increases the recall of the model. This same pattern can be seen repeated if we look at the overall macro avg precision and recall of the model. Precision is defined as the total number of identified true positives divided by the total number of records labelled as positive by the model. Recall is defined as number of identified true positives divided by the total number of actual records that actually belong to the positive class.

Keeping these definitions in mind, it makes sense that the recall for the `CHANGE` class increases when models are trained with equalized class counts, since then it identifies a higher number of true positives from the total set of true positives. Naturally, as the recall increases when trained on equalized class counts, the precision goes down (since these two metrics counter each other).

As per our understanding, the precision being higher for the `CHANGE` class for models trained on un-equalized class counts data also makes sense, as these models (being overfit to a certain extent on the `SAME` class) make lesser predictions for the `CHANGE` class, and as this class has a lesser amount of records in the testing data anyway, the precision is higher.

It also stands to be seen that using attention generally gives better results for most metrics (but not necessarily all!).

Ultimately, identifying which model is better depends upon the scenario in question. For example, if identifying as many records of the `CHANGE` class as possible is more important for us, we should choose the models with a higher recall for this class. 

## Scripts

We have the following scripts in use:

- `rnn.py`: This script contains the class definition for the RNN network, and the classifier network without attention.
- `attention_rnn.py`: This script contains the class definition for the RNN network, the attention network, and the classifier network with attention.
- `dataset_updated.py`: This script contains the Dataset class to be used by the dataloader.
- `read_data.py`: This script loads the files from the path specified in the config file, applies the pre-processing techniques on them as discussed previously, performs a train-test split on the data (default 80-20 if the user doesn't specify something else), and then writes the training and testing data to individual files (the names of which are specified in the config file).
- `utils.py`: This script contains a bunch of utility functions which are used by the various training and testing scripts.
- `config.py`: This script contains various configurations ranging from file paths and model configurations to model names, whether to equalize class counts or not, whether to use attention or not etc.
- `train_rnn_updated.py`: This is the training script that is used for training the RNN models and classifier. This same script can be used to train a model without attention and without equalized class counts, without attention and with equalized class counts, with attention and without equalized class counts, and with attention and with equalized class counts. These configurations can be set in the config file, and then this file can be run to train the models and classifier and save them to disk.
- `test_rnn_updated.py`: This script will load the model from disk as per the configuration specified in the config file, and then use it to make predictions on the records found in the testing file.
- `train_bert_model.py`: This script loads a BERT model and fine-tunes it as per the data in the training file, and saves the model and tokenizer to disk.
- `test_bert_model.py`: This script loads the fine-tuned BERT model and tokenizer saved to disk (the paths to which is specified in the config file), and uses it to make predictions on the records found in the testing file.

## Bonuses

### Another method

The third model, RNN with attention mechanism, was used for this bonus part. The motivation behind using this approach has been described earlier, when explaining the network.

## Future work

The first thing we would want to try for future work would be to apply the attention mechanism on the word level (second dimension of the output vector) instead of the hidden size dimension (third dimension of the output vector). It would be interesting to see if that has a noticeable impact on the results, and potentially help us identify if we're applying attention correctly in this case or not.

Another thing that would be nice to do would be to see if we can oversample the data for the `CHANGE` class, and then potentially be able to train the model on a much larger amount of data while retaining equality for the amount of data for both classes. We briefly looked into oversampling techniques, and they seem to be geared more towards numeric data rather than text data - perhaps we could apply them at the word index or word vector level somehow.