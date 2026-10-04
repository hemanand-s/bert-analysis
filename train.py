import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

from config import (
    MODEL_NAME,
    MODEL_DIR,
    OUTPUT_DIR,
    TRAIN_FILE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    SEED
)
from preprocess import load_data
from dataset import SentimentDataset


def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            labels, preds, average="weighted", zero_division=0
        )
    }


def main():

    set_seed(SEED)

    train_encodings, test_encodings, train_labels, test_labels, tokenizer = load_data(
        TRAIN_FILE
    )

    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)

    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = np.asarray(test_labels)
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    print("\n----- Evaluation Metrics -----")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))

    print("\n----- Classification Report -----")
    print(classification_report(y_true, y_pred, target_names=label_names))

    print("----- Confusion Matrix -----")
    print(confusion_matrix(y_true, y_pred))

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"\nModel saved to {MODEL_DIR}")
    print("Training Completed")


if __name__ == "__main__":
    main()