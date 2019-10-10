from torch import cuda


data_dir = "hansard/scrapedxml/debates/"
CSV_FILENAME = "parliamentary_speech_data.csv"
PATH_TO_PRETRAINED_EMBEDDINGS = '/home/azfar/PythonProjects/GoogleNews-vectors-negative300.bin.gz'

BOUNDARY_TO_INT_MAPPING = {
    '[SAME]': 0,
    '[CHANGE]': 1
}

DEVICE = "cuda:1" if cuda.is_available() else "cpu"
BERT_BATCH_SIZE = 32
BERT_NUM_EPOCHS = 4
