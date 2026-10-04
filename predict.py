from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bert_sentiment_model",
    tokenizer="bert_sentiment_model"
)

text = input("Enter Review: ")

result = classifier(text)

label = result[0]["label"]

if label == "LABEL_0":
    sentiment = "Negative"
else:
    sentiment = "Positive"

print("\nPredicted Sentiment:", sentiment)