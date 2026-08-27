# Hale Kannada to Modern Kannada

An NLP-based machine translation project for converting **Hale Kannada (Old Kannada)** into **Hosa Kannada (Modern Kannada)** using a fine-tuned IndicTrans2 model.

## Project Overview

Kannada has evolved significantly over time. Historical Kannada texts often use vocabulary, grammar, and linguistic structures that differ from modern Kannada.

The goal of this project is to develop an AI-based system that can take **Hale Kannada text as input** and generate its corresponding **Modern Kannada (Hosa Kannada)** form.

### Translation Pipeline

```text
Hale Kannada
     ↓
Data Preprocessing
     ↓
IndicTrans2 320M
     ↓
Fine-tuning
     ↓
Modern Kannada
```

## Current Progress

* [x] Project repository created
* [x] Hale Kannada → Hosa Kannada dataset prepared
* [x] Dataset divided into training, validation, and test sets
* [x] Original dataset added to the project
* [ ] Dataset cleaning and preprocessing
* [ ] Model fine-tuning
* [ ] Model evaluation
* [ ] Translation testing
* [ ] Web application integration

## Dataset

The project uses a parallel dataset containing corresponding Hale Kannada and Modern Kannada sentences.

The dataset is divided into:

```text
original_dataset/
│
├── train.csv
├── validation.csv
└── test.csv
```

### Dataset Splits

* `train.csv` — Used for model training
* `validation.csv` — Used for monitoring and tuning the model during training
* `test.csv` — Used for final evaluation

## Dataset Preparation

The dataset has been prepared specifically for the **Hale Kannada → Modern Kannada** translation task.

The preprocessing stage will include:

* Cleaning unnecessary text
* Handling missing values
* Removing duplicate sentence pairs
* Normalizing the text
* Checking sentence alignment
* Preparing the data for model fine-tuning

The cleaned datasets will be stored separately from the original dataset.

```text
original_dataset/
    ↓
Dataset Preparation
    ↓
cleaned_datasets/
```

## Base Model

The project uses the following pretrained model as the starting point:

```text
ai4bharat/indictrans2-indic-indic-dist-320M
```

IndicTrans2 is a multilingual machine translation model developed for Indian languages.

The pretrained model will be fine-tuned using the prepared Hale Kannada → Modern Kannada parallel dataset.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* IndicTransToolkit
* Pandas
* IndicTrans2
* Git
* GitHub

## Project Structure

```text
hale-to-hosa-kannada/
│
├── original_dataset/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── cleaned_datasets/
│
├── dataset.py
├── model.py
├── train.py
├── test.py
├── app.py
├── index.html
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Some files and folders will be added as the project development progresses.

## Development Workflow

The project is developed incrementally using Git and GitHub.

Each major stage is committed separately:

```text
Dataset Preparation
        ↓
Data Cleaning
        ↓
Model Preparation
        ↓
Fine-tuning
        ↓
Evaluation
        ↓
Testing
        ↓
Web Application
```

## Future Work

* Clean and preprocess the dataset
* Fine-tune IndicTrans2 on the Hale Kannada → Modern Kannada dataset
* Evaluate translation quality
* Test the model on unseen Hale Kannada sentences
* Improve translation quality through hyperparameter tuning
* Build a web-based translation interface
* Deploy the application

## Author

**Ajay Kumar Y R**

GitHub: [Ajay-kumar-yr](https://github.com/Ajay-kumar-yr)
