MODEL_NAME = "bert-base-uncased"
MODEL_DIR = "bert_sentiment_model"
OUTPUT_DIR = "./results"

TRAIN_FILE = "sentiment.csv"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
SAMPLE_SIZE = 50000
SEED = 42

NUM_LABELS = 2

LABEL_MAP = {
    0: 0,
    4: 1
}

ID2LABEL = {
    0: "negative",
    1: "positive"
}

LABEL2ID = {
    "negative": 0,
    "positive": 1
}