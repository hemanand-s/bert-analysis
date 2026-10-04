import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from config import (
    MODEL_NAME,
    MAX_LENGTH,
    LABEL_MAP,
    TEST_SIZE,
    SAMPLE_SIZE,
    SEED,
    TRAIN_FILE
)


def load_data(file_path=TRAIN_FILE):

    df = pd.read_csv(
        file_path,
        encoding='latin1',
        header=None,
        names=['sentiment', 'id', 'date', 'query', 'user', 'text']
    )

    df = df[df['sentiment'].isin(LABEL_MAP)].copy()

    df['label'] = df['sentiment'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=SEED)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        list(train_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    test_encodings = tokenizer(
        list(test_texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    return (
        train_encodings,
        test_encodings,
        train_labels,
        test_labels,
        tokenizer
    )