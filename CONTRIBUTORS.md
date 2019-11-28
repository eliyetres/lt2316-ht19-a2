# Description of team member contribution

## Group

### Azfar Imtiaz

- Help out with data reading tasks, write data to pickle files (obsolete now)
- Write function to get vectors for words from the Word2Vec model (with support for padding and unknown tokens)
- Write function to generate sentence vectors
- Write function to generate batch vectors at train/test time
- Write script to fine-tune the BERT model (using the `pytorch_pretrained_bert` module) on the training data
- Fix bug of mismatch between maximum sentence length and number of epochs used for training BERT
- Save BERT model and tokenizer to disk, using the rather convoluted method of saving the model as .bin file, saving the config file, and packaging them into a .tar.gz file for the former
- Write script for evaluating the BERT model on testing data
- Correct and improve structure for RNN network
- Combine RNN hidden states and feed them into classifier network
- Write the updated script for training the RNN models and classifier model on training data
- Create RNN network, attention network and classifier network for bonus part
- Incorporate support for attention model in the updated training script for RNN
- Fix bugs in the prediction part of the RNN and classifier models
- Add support for equalizing class counts in training data
- Add support for using an additional amount of records for the `SAME` class in training data
- Update config file to contain all sorts of parameters for model training and testing, such as whether to use attention or not, whether to use equalized class counts in training or not, batch size, hidden size etc for all three model types.
- Train all models on 100,000 records of training data, evaluate them on the testing data, and report results to be put into the README
- Add details into README about the network structure, motivation behind using attention mechanism, and information about the function of each script
- Bug fix for using command line arguments, add check to ensure that the test size ratio provided is between 0.0 and 1.0

### Elin Hagman

- Reading and parsing XML file, determine speaker names and boundaries
- Replacing British words and compound words to their American equivalent
- Removing weird tokens
- Writing sentences and boundaries to .csv file
- Optimized training script and RNN models for GPU
- Created testing script for RNN models and classifier
- Created testing script for RNN models and models with attention
- Documented all changes and decisions in report
- Documented results of trained models

### Sandra Derbring

- Creating a dataloader for the dataset
- First version of train test splitter
- Pre processing data, changing abbreviations to full words
- Post processing script, check tokenization and concatenate words
- Built structure of first RNN network model
- First training script for RNN network model
- Enabling command line options for the train/test splitting