# LT2316 HT19 Assignment 2

Sentence pair classification for parliamentary turn-taking
From Asad Sayeed's Machine Learning course at the University of Gothenburg.

## Group Broccoli 🥦

Azfar Imtiaz\
Elin Hagman\
Sandra Derbring

## Data processing

### Pre processing

In the pre processing, the paragraphs are first selected by their speech tags and from that the speaker names are extracted. The speeches for the "nospeaker" which is the parliament's President speaking are also included. Then for every sentence, the  boundary SAME was added to all sentences when the speaker does not change. For every new speech, a list is kept of the current and previous speaker to check if it's they're the same, and popping the list for each new speaker name in the speech tag. If the speaker changes, the boundary is set to NEW, if not it stays SAME. The text is  saved to a csv file using a pandas object and compressed to a .rar file because this was the smallest size that was manageable.
The sentences and boundaries were written to a csv file in the format: \
Sentence1|Sentence2| Boundary

### Post processing

Words like "hon." are changed to their full word equivalent "honorary" to make sure they match their correct word vector in the gensim model. Some British spelled words were changed to their American equivalent, and hyphens were removed for compound words that were spelled with hyphens in British English but not in American. The rest of the words not found in the gensim model at all were removed from the text. Removing some words resulted in some sentences just being an empty string so in the end, these sentences were also removed after making sure that not too much data was lost.

## Training and test data

One problem noticed during training was that there were a lot more records for SAME as opposed to CHANGE.
This was solved by adding the parameter RNN_SAME_ADDITIONAL_RECORDS to the config file.  Using the parameter its possible to specify how many additional records for the SAME category to select. For example, if RNN_NUM_RECORDS is 10000, that means 5000 for SAME and 5000 for CHANGE. Moreover, if  RNN_SAME_ADDITIONAL_RECORDS=1500, it means there is 5000 for CHANGE but 6500 now for SAME. This leaves us with slightly more records for SAME as opposed to CHANGE. The train- and test data is split into train and testing and saved to two separate csv files which are loaded in the train- and testing scripts. The train data size it the conventional 80% for training data and 20% for test data, and its possible to change this using a parameter in the training script.

## Approach

### RNN network- and classifier models

Two models of the same Recurrent network class are created with two optimizers. The criterion used is Binary Cross Entropy, since the possible labels are either 0 or 1 (SAME or NEW). The first sentence is fed into one object, and the second sentence into the other object.  The batches are fetched for both training generators simultaneously, creating the hidden states after the forward pass of both models. The output from both these models are combined by concatenating, e.g. if the output is 300-dimensional and there are two outputs, this would result in a 600-dimensional vector which is fed into another network that contains a linear layer followed by a Sigmoid. This final network is used for the actual classification. Initially two LSTMs were used but was changed to GRU since it returns a single hidden state and is easier to deal with. Some structural changes were also made from the initial RNN, it was concluded to not be a good idea to have the optimizer and criterion inside the class, as that would make the weights not update properly.

### RNN network model with Attention

The idea of adding an attention mechanism to the RNN network is that attention should help the classifier identify which parts of the concatenated output are important for making the prediction. Ideally, that should be words that signify a change or sameness, like "My honour.". If the first sentence includes these words, it's almost always the same speaker (SAME label).
The models are created from the same classes as the RNN and classifier models. It passes the concatenated output states of the two RNNs and then applies softmax to them to get probabilities. Then the probabilities are multiplied with the concatenated output to get a weighted output, and is then fed into the linear layer for classification.

### Bert network model

The pre-processed text that was initially saved as a csv is loaded, so the test data is processed the same way as the training data. This helps avoid having to run the preprocess_data script multiple times. Further data processing is made to fit the Bert model. Tokenized sentences are clipped to fit the max size, as well as creating a segment mask for the training data. The segment mask is used to identify whether the input is one sentence or two sentences long. If the model uses one sentence, the mask is simply a sequence of 0s. For two sentence inputs, there is a 0 for each token of the first sentence, followed by a 1 for each token of the second sentence. The labels are binary, and formatted the same way as the RNN model, using 0 or 1.

## Evaluation

### RNN evaluation

The text is processed in the same way as for the training data. As opposed to to multi-class classification where the class with the highest probability after applying softmax is selected, the prediction for binary classification is computed by getting the probability and check if it's >=0.5 and assign it to class 1, assign it to class 0 otherwise.  The labels for the correct class and the predicted are saved and used to generate a classification report showing precision, recall and f1-score using Sklearn's classification report.

### Results

#### RNN trained without attention and without equalized class counts  

Total length of training data: 100000\
Number of SAME records: 88000\
Number of CHANGE records: 12000

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

Total length of training data: 110000\
Number of SAME records: 60000\
Number of CHANGE records: 50000

## Bonuses

### Another method

The third model, RNN with attention mechanism was used for this bonus part.
