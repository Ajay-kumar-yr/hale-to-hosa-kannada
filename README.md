# Hale Kannada to Modern Kannada

An NLP-based machine translation project for converting **Hale Kannada (Old Kannada)** into **Hosa Kannada (Modern Kannada)** using a fine-tuned **IndicTrans2** model.

## Project Overview

Kannada has evolved significantly over time. Historical Kannada texts often contain vocabulary, grammatical structures, and linguistic forms that differ from modern Kannada.

The goal of this project is to develop an AI-based translation system that takes **Hale Kannada text as input** and generates the corresponding **Hosa Kannada (Modern Kannada)** form.

### Translation Pipeline

```text
Hale Kannada
     ↓
Dataset Preparation
     ↓
Data Cleaning & Preprocessing
     ↓
IndicTrans2 320M
     ↓
Fine-tuning
     ↓
Hosa Kannada
```

## Current Progress

* [x] GitHub repository created
* [x] Hale Kannada → Hosa Kannada dataset prepared
* [x] Dataset divided into training, validation, and test sets
* [x] Original dataset added to the project
* [x] Dataset cleaning and preprocessing completed
* [x] Fine-tuning pipeline implemented
* [x] 100-example test fine-tuning completed successfully
* [x] Validation pipeline tested
* [ ] Full 27K+ dataset fine-tuning
* [ ] BLEU / automatic evaluation
* [ ] Translation quality evaluation
* [ ] Complete web application
* [ ] Deployment

## Dataset

The project uses a parallel dataset containing corresponding Hale Kannada and Hosa Kannada text pairs.

### Original Dataset

```text
datasets/
└── original_dataset/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

### Dataset Splits

* `train.csv` — Training data
* `validation.csv` — Validation data
* `test.csv` — Test data

The cleaned dataset contains:

```text
Training examples      : 27,112
Validation examples    : 2,921
```

The test dataset is reserved for final evaluation.

## Dataset Preparation

The dataset was cleaned and prepared using the preprocessing scripts in the project.

### Preprocessing Steps

* Loaded the original training, validation, and test datasets
* Filtered relevant translation tasks
* Extracted Hale Kannada source text
* Extracted Hosa Kannada target text
* Removed invalid and empty rows
* Removed duplicate entries
* Normalized text formatting
* Generated clean parallel datasets

### Cleaned Dataset

The processed datasets are stored in:

```text
datasets/
└── cleaned_datasets/
    ├── train_final.csv
    ├── validation_final.csv
    └── test_final.csv
```

Each cleaned CSV contains two columns:

```text
source,target
```

Example:

```text
source,target
ಅಂಕಂಗೊಳ್,ಅಂಕಂಗುಡು
```

## Base Model

The project uses:

```text
ai4bharat/indictrans2-indic-indic-dist-320M
```

IndicTrans2 is a multilingual neural machine translation model developed for Indian languages.

The pretrained model is used as the base model and is fine-tuned on the Hale Kannada → Hosa Kannada parallel dataset.

## IndicTrans2 Language Format

Both Hale Kannada and Hosa Kannada use the Kannada script.

For IndicTrans2, the Kannada language code is:

```text
kan_Knda
```

During tokenization, the model expects the format:

```text
kan_Knda kan_Knda <input text>
```

For example:

```text
kan_Knda kan_Knda ಅಂಕಂಗೊಳ್
```

The dataset itself does **not** need to contain these language prefixes. They are added during preprocessing before tokenization.

## Fine-tuning

The fine-tuning pipeline is implemented in:

```text
fine_tune.py
```

The training uses:

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Seq2SeqTrainer
* IndicTrans2 320M

### Training Configuration

The current configuration includes:

```text
Maximum source length : 128 tokens
Maximum target length : 128 tokens
Learning rate         : 5e-5
Batch size             : 1
Gradient accumulation  : 8
```

The exact training configuration can be found in `fine_tune.py`.

## Test Fine-tuning

Before running the complete training process, a small test run was performed.

```text
Training examples   : 100
Validation examples : 20
Epochs              : 1
```

The test run successfully completed:

```text
Dataset loading
      ↓
Tokenization
      ↓
Model loading
      ↓
Fine-tuning
      ↓
Validation
      ↓
Model saving
```

This confirms that the complete training pipeline is functioning correctly.

## GPU Training

The local development machine uses:

```text
GPU: Intel UHD Graphics
CUDA: Not available
```

Therefore, full model fine-tuning is planned to be performed using a cloud GPU environment such as **Google Colab**.

The training script automatically detects CUDA:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

GPU training is recommended for the complete dataset because CPU training is significantly slower.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* IndicTrans2
* Pandas
* NumPy
* Flask
* Git
* GitHub

> **Note:** `IndicTransToolkit` is not required by the current fine-tuning pipeline.

## Installation

### Prerequisites

* Python 3.8+
* pip
* Git
* GPU recommended for full training

### Clone Repository

```bash
git clone https://github.com/Ajay-kumar-yr/hale-to-hosa-kannada.git

cd hale-to-hosa-kannada
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```text
hale_hosa_kannada/
│
├── datasets/
│   │
│   ├── original_dataset/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   └── cleaned_datasets/
│       ├── train_final.csv
│       ├── validation_final.csv
│       └── test_final.csv
│
├── prepare_data.py
├── fine_tune.py
├── model.py
├── app.py
├── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

The fine-tuned model directory:

```text
hale_hosa_model/
```

is intentionally excluded from Git using `.gitignore` because model weights are large.

## Running Fine-tuning

The fine-tuning script can be started with:

```bash
python fine_tune.py
```

### Test Mode

For testing the pipeline with a small dataset:

```python
TEST_MODE = True
```

This uses a small number of training and validation examples.

### Full Training

After confirming that the test run works:

```python
TEST_MODE = False
```

The full training dataset can then be used.

For full training, a CUDA-enabled GPU environment such as Google Colab is recommended.

## Inference

The fine-tuned model can be loaded using the Hugging Face Transformers library.

Example:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

MODEL_PATH = "hale_hosa_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = model.to(DEVICE)
model.eval()

text = "ಅಂಕಂಗೊಳ್"

input_text = f"kan_Knda kan_Knda {text}"

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=128,
    truncation=True
)

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}

with torch.inference_mode():

    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=128
    )

result = tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True
)

print("Hale Kannada :", text)
print("Hosa Kannada :", result[0])
```

> The inference code may be updated as the fine-tuned model and evaluation pipeline evolve.

## Web Application

A Flask-based web interface is being developed to provide a simple single-page translation interface.

Current planned architecture:

```text
User
  ↓
Web Interface
  ↓
Flask
  ↓
Fine-tuned IndicTrans2
  ↓
Hosa Kannada
```

### Current Status

* [ ] Complete frontend
* [ ] Connect frontend to model
* [ ] REST API
* [ ] Translation endpoint
* [ ] Deployment

## Development Workflow

The project is developed incrementally using Git and GitHub.

```text
Dataset Preparation
        ↓
Data Cleaning
        ↓
Model Preparation
        ↓
Test Fine-tuning
        ↓
Full Fine-tuning
        ↓
Evaluation
        ↓
Translation Testing
        ↓
Web Application
        ↓
Deployment
```

### Git Workflow

After making changes:

```bash
git status
git add .
git commit -m "Describe your changes"
git push
```

## Model Evaluation

The following evaluation methods are planned:

* BLEU score
* Test-set evaluation
* Manual translation evaluation
* Example-based qualitative analysis
* Comparison between base and fine-tuned models

Final evaluation results will be added after full model training.

## Future Work

* [ ] Full fine-tuning on the complete dataset
* [ ] BLEU evaluation
* [ ] Additional translation metrics
* [ ] Manual linguistic evaluation
* [ ] Improve translation quality
* [ ] Complete web application
* [ ] Build REST API
* [ ] Deploy translation service
* [ ] Batch translation
* [ ] Model optimization
* [ ] Model quantization
* [ ] Expand the dataset
* [ ] Support additional historical Kannada forms

## Author

**Ajay Kumar Y R**

GitHub: https://github.com/Ajay-kumar-yr
