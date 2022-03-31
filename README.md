# MuCoT - Multilingual Contrastive Training Approach for Low-resource Languages

> Accepted at [ACL'22](https://www.2022.aclweb.org/) [Workshop for Dravidian Language Technologies](https://dravidianlangtech.github.io/2022/)

Question Answering (QA) in English has improved a lot in recent years with the adaption of BERT-based models. These models are pre-trained in a self-supervised fashion with a huge English text corpus and further fine-tuned with a massive English QA dataset: SQuAD. However, QA dataset on such a scale is not available for most of the other languages. Thus, the multi-lingual BERT-based models, are used to transfer knowledge from the high resource languages to low resource languages. These models are pre-trained with huge text corpora in multiple languages and typically process tokens from multiple languages into language agnostic embeddings. In this work using Google's ChAII dataset, we empirically show that fine-tuning multi-lingual BERT-based models with translations from the same language family boost the question-answering performance whereas it degrades the performance in case of cross-language families. Further, we show that introducing contrastive loss, between the translated question-context feature pairs in the fine-tuning process, prevents such a fall with cross-lingual family translations or improves it.

**Authors**: Gokul Karthik Kumar, Abhishek Singh Gehlot, Sahal Shaji Mullappilly, Karthik Nandakumar

**Full Paper**: [Link](#) (will be updated)

# Idea
<img src='images/mucot.png' width=512>

<img src='images/clip.png' width=512>

# Results
<img src='images/result_1.png' width=512>

## Datasets

### Chaii

* Chaii Original - [Kaggle Link](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data)
* Chaii Translated & Transliterated - [Kaggle Link](https://www.kaggle.com/gokulkarthik/chaiitrans)

> Run [data/chaii_split.py](./data/chaii_split.py) from the root directory make the train-val-test splits.
<img src='images/chaii_dataset_info.png' width=512>

## Models

### Pretrained:
- [mBERT](https://huggingface.co/bert-base-multilingual-cased) | [mBERT SQUAD](https://huggingface.co/salti/bert-base-multilingual-cased-finetuned-squad)

