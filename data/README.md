# Dataset

## Emotion Dataset for NLP

This project uses the **Emotion Dataset for NLP** by Elvis Saravia et al.

- **Source:** [Kaggle — Emotion Dataset for NLP](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
- **Paper:** Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018
- **Size:** 20,000 labeled English tweets/short texts
- **Classes:** joy, sadness, anger, fear, love, surprise (6 classes)

## Download

```bash
# Using Kaggle API
kaggle datasets download -d parulpandey/emotion-dataset
unzip emotion-dataset.zip -d data/
```

## Splits

| File | Samples |
|------|--------:|
| train.txt | 16,000 |
| val.txt | 2,000 |
| test.txt | 2,000 |

## Class Distribution (Training Set)

| Emotion | Count | % |
|---------|------:|--:|
| Joy | 5,362 | 33.5% |
| Sadness | 4,666 | 29.2% |
| Anger | 2,159 | 13.5% |
| Fear | 1,937 | 12.1% |
| Love | 1,304 | 8.2% |
| Surprise | 572 | 3.6% |

> The dataset is **not included** in this repository due to licensing.  
> Please download it directly from Kaggle using the link above.
