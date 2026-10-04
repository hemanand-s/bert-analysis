import sys

from transformers import pipeline

from config import MODEL_DIR

classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    truncation=True
)

if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    text = input("Enter Review: ")

if not text.strip():
    sys.exit("No text provided.")

result = classifier(text)[0]

print("\nPredicted Sentiment:", result["label"].capitalize())
print("Confidence         : {:.2%}".format(result["score"]))