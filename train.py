from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from preprocess import load_data
from dataset import SentimentDataset
from config import MODEL_NAME, EPOCHS, BATCH_SIZE

# Load Dataset
train_encodings, test_encodings, train_labels, test_labels, tokenizer = load_data(
    "sentiment.csv"
)

# Dataset Objects
train_dataset = SentimentDataset(
    train_encodings,
    train_labels
)

test_dataset = SentimentDataset(
    test_encodings,
    test_labels
)

# Load Model
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train
trainer.train()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get predictions on test data
predictions = trainer.predict(test_dataset)

y_pred = predictions.predictions.argmax(axis=1)
y_true = test_labels.to_numpy()

print("\n----- Evaluation Metrics -----")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='weighted'))
print("Recall   :", recall_score(y_true, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_true, y_pred, average='weighted'))

# Save
model.save_pretrained(
    "bert_sentiment_model"
)

tokenizer.save_pretrained(
    "bert_sentiment_model"
)

print("Training Completed")
