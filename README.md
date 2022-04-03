# MuCoT - Multilingual Contrastive Training Approach for Low-resource Languages

<div align="center">
    <a href="https://jupyter.org/">
        <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Jupyter%20Notebook-informational?style=flat-sqaure&logo=jupyter&logoColor=black&color=F37626">
    </a>
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/Python-informational?style=flat-sqaure&logo=python&logoColor=white&color=3776AB">
    </a>
    <br>
    <br>
</div>

> 🎉 Accepted for oral presentation at [#ACL 2022](https://www.2022.aclweb.org/) [Workshop for Dravidian Language Technologies](https://dravidianlangtech.github.io/2022/)

Question Answering (QA) in English has improved a lot in recent years with the adaption of BERT-based models. These models are pre-trained in a self-supervised fashion with a huge English text corpus and further fine-tuned with a massive English QA dataset: SQuAD. However, QA dataset on such a scale is not available for most of the other languages. Thus, the multi-lingual BERT-based models, are used to transfer knowledge from the high resource languages to low resource languages. These models are pre-trained with huge text corpora in multiple languages and typically process tokens from multiple languages into language agnostic embeddings. In this work using Google's ChAII dataset, we empirically show that fine-tuning multi-lingual BERT-based models with translations from the same language family boost the question-answering performance whereas it degrades the performance in case of cross-language families. Further, we show that introducing contrastive loss, between the translated question-context feature pairs in the fine-tuning process, prevents such a fall with cross-lingual family translations or improves it.

**TL;DR:** We use contrastive loss between the translated pairs during fine-tuning to improve multilingual BERT for question answering in low-resource languages.

**Authors**: Gokul Karthik Kumar, Abhishek Singh Gehlot, Sahal Shaji Mullappilly, Karthik Nandakumar

**Full Paper**: [Link](#) (will be updated)

---

# Index
- [Idea](#idea)
- [Results](#results)
- [Datasets](#datasets)
  - [Chaii](#chaii)
- [Models](#models)
  - [Pretrained](#pretrained)

---
<br>

# Idea
<img src='images/mucot.png' width=1024>

<img src='images/clip.png' width=1024>

# Results
<img src='images/result_1.png' width=1024>

## Datasets

### Chaii

<!-- * Chaii Original - [Kaggle Link](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data)
    * Chaii Translated & Transliterated - [Kaggle Link](https://www.kaggle.com/gokulkarthik/chaiitrans) -->
<ul>
    <li>
        Chaii Original -
        <a href="https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data">
            <img alt="Kaggle" title="Click here to view the dataset" src="https://img.shields.io/badge/Kaggle-informational?style=flat-sqaure&logo=kaggle&logoColor=black&color=20BEFF">
        </a>
    </li>
    <li>
        Chaii Translated & Transliterated -
            <a href="https://www.kaggle.com/gokulkarthik/chaiitrans">
            <img alt="Kaggle" title="Click here to view the dataset" src="https://img.shields.io/badge/Kaggle-informational?style=flat-sqaure&logo=kaggle&logoColor=black&color=20BEFF">
        </a>
    </li>
</ul>

> Run [`data/chaii_split.py`](./data/chaii_split.py) from the root directory make the train-val-test splits.
<img src='images/chaii_dataset_info.png' width=512>

## Models

### Pretrained:
- [mBERT](https://huggingface.co/bert-base-multilingual-cased) | [mBERT SQUAD](https://huggingface.co/salti/bert-base-multilingual-cased-finetuned-squad)

