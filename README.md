# BERT Sentiment Analysis

Fine-tunes `bert-base-uncased` for binary sentiment classification on
[Sentiment140](https://kaggle.com/datasets/krishnarpit7/sentiment140), which labels
tweets as negative (`0`) or positive (`4`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `sentiment.csv` in the project root. It is ignored by git because it is
~228 MB, so download it manually from Kaggle.

## Train

```bash
python train.py
```

Tunes are in `config.py`. Defaults sample 50,000 tweets and train for 3 epochs,
which takes roughly 15-25 minutes on a 6 GB GPU. Set `SAMPLE_SIZE` to `0` to use
the full dataset.

The run prints accuracy, precision, recall, weighted F1, a per-class
classification report and a confusion matrix, then writes the model and tokenizer
to `bert_sentiment_model/`.

## Predict

```bash
python predict.py
python predict.py "the battery life on this phone is terrible"
```

```
Predicted Sentiment: Negative
Confidence         : 98.40%
```

The model is trained with `id2label`, so predictions come back as `negative` and
`positive` rather than `LABEL_0` and `LABEL_1`.

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | All tunable hyperparameters and the label maps |
| `preprocess.py` | Reads the CSV, maps labels, splits, tokenizes |
| `dataset.py` | Torch `Dataset` wrapping the tokenized encodings |
| `train.py` | Fine-tuning loop, evaluation and model export |
| `predict.py` | Inference on a single piece of text |

## Notes

`model.safetensors` (~418 MB) and `sentiment.csv` (~228 MB) exceed GitHub's
100 MB per-file limit, so both are gitignored and stay local.