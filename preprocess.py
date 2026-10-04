import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from config import MODEL_NAME, MAX_LENGTH

def load_data(file_path):

    df =pd.read_csv(
    "sentiment.csv",
    encoding='latin1',
    header=None,
    names=['sentiment','id','date','query','user','text']
)

# Use only 5000 samples
    df = df.sample(5000, random_state=42)

    # Convert Sentiment140 labels
    sentiment_map = {
        0: 0,   # Negative
        2: 1,   # Neutral (if present)
        4: 2    # Positive
    }

    df = df[df['sentiment'].isin([0, 4])]

    df['label'] = df['sentiment'].map({
        0: 0,
        4: 1
    })

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42
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