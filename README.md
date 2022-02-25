# multilingual-qa

## Datasets

### Chaii

* Chaii Original - [Kaggle Link](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data)
* Chaii Translated & Transliterated - [Kaggle Link](https://www.kaggle.com/gokulkarthik/chaiitrans)

> Run [data/chaii_split.py](./data/chaii_split.py) from the root directory make the train-val-test splits.
<img src='images/chaii_dataset_info.png' width=512>

## Models

### Pretrained:
- [mBERT](https://huggingface.co/bert-base-multilingual-cased) | [mBERT SQUAD](https://huggingface.co/salti/bert-base-multilingual-cased-finetuned-squad)
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) | [XLM-RoBERTa SQUAD](https://huggingface.co/deepset/xlm-roberta-base-squad2)
- [Distill-BERT](https://huggingface.co/distilbert-base-multilingual-cased)
- [MuRIL](https://huggingface.co/google/muril-base-cased)
- [Indic-BERT](https://huggingface.co/ai4bharat/indic-bert)
