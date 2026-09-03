import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_datasets",
    "train_final.csv",
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_datasets",
    "validation_final.csv",
)

# Model output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "hale_hosa_model",
)


# ============================================================
# 2. TRAINING SETTINGS
# ============================================================

# True  -> small test run
# False -> full training
TEST_MODE = True

# Used only when TEST_MODE = True
TEST_TRAIN_SIZE = 100
TEST_VALIDATION_SIZE = 20

# Number of epochs for full training
FULL_TRAIN_EPOCHS = 3

# Maximum token lengths
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

# Training parameters
TRAIN_BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2

# Effective batch size:
# 2 * 4 = 8
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


# ============================================================
# 3. GPU INFORMATION
# ============================================================

print("=" * 60)
print("Hale Kannada -> Hosa Kannada Fine-Tuning")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )
    DEVICE = "cuda"
else:
    print("WARNING: CUDA GPU not detected.")
    print("Training will run on CPU and may be very slow.")
    DEVICE = "cpu"

print("Device:", DEVICE)


# ============================================================
# 4. CHECK DATASET FILES
# ============================================================

print("\nChecking dataset files...")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

print("Training file:", TRAIN_FILE)
print("Validation file:", VALIDATION_FILE)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Original training shape:", train_df.shape)
print("Original validation shape:", validation_df.shape)

# Check required columns
required_columns = {"source", "target"}

if not required_columns.issubset(train_df.columns):
    raise ValueError(
        f"Training dataset must contain columns: {required_columns}"
    )

if not required_columns.issubset(validation_df.columns):
    raise ValueError(
        f"Validation dataset must contain columns: {required_columns}"
    )


# ============================================================
# 6. CLEAN DATA
# ============================================================

train_df = train_df[["source", "target"]].copy()
validation_df = validation_df[["source", "target"]].copy()

train_df["source"] = train_df["source"].fillna("").astype(str).str.strip()
train_df["target"] = train_df["target"].fillna("").astype(str).str.strip()

validation_df["source"] = (
    validation_df["source"].fillna("").astype(str).str.strip()
)

validation_df["target"] = (
    validation_df["target"].fillna("").astype(str).str.strip()
)

# Remove empty rows
train_df = train_df[
    (train_df["source"] != "") &
    (train_df["target"] != "")
].reset_index(drop=True)

validation_df = validation_df[
    (validation_df["source"] != "") &
    (validation_df["target"] != "")
].reset_index(drop=True)

print("Training rows after cleaning:", len(train_df))
print("Validation rows after cleaning:", len(validation_df))


# ============================================================
# 7. TEST MODE
# ============================================================

if TEST_MODE:

    print("\n" + "=" * 60)
    print("TEST MODE ENABLED")
    print("=" * 60)

    train_df = train_df.head(TEST_TRAIN_SIZE)
    validation_df = validation_df.head(TEST_VALIDATION_SIZE)

    print("Test training examples:", len(train_df))
    print("Test validation examples:", len(validation_df))

else:

    print("\n" + "=" * 60)
    print("FULL TRAINING MODE ENABLED")
    print("=" * 60)

    print("Training examples:", len(train_df))
    print("Validation examples:", len(validation_df))


# ============================================================
# 8. CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    validation_df,
    preserve_index=False,
)

print("\nHugging Face datasets created successfully.")


# ============================================================
# 9. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Tokenizer loaded successfully.")


# ============================================================
# 10. LOAD MODEL
# ============================================================

print("\nLoading IndicTrans2 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model.to(DEVICE)

print("Model loaded successfully.")
print("Model device:", next(model.parameters()).device)


# ============================================================
# 11. TOKENIZATION
# ============================================================

def preprocess_function(examples):

    # IndicTrans2 language format:
    # source language + target language + source text
    #
    # Both source and target are Kannada:
    # kan_Knda -> kan_Knda

    source_texts = [
        f"kan_Knda kan_Knda {text}"
        for text in examples["source"]
    ]

    target_texts = examples["target"]

    model_inputs = tokenizer(
        source_texts,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=target_texts,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing training dataset...")

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print("Tokenizing validation dataset...")

tokenized_validation = validation_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=validation_dataset.column_names,
)

print("Tokenization completed.")


# ============================================================
# 12. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# 13. TRAINING ARGUMENTS
# ============================================================

num_epochs = 1 if TEST_MODE else FULL_TRAIN_EPOCHS

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    overwrite_output_dir=False,

    num_train_epochs=num_epochs,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALIDATION_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,

    # Checkpoints
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    # Logging
    logging_strategy="steps",
    logging_steps=20,

    # GPU
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=False,

    # Data loading
    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    # Do not generate translations during evaluation.
    # This makes evaluation considerably faster.
    predict_with_generate=False,

    # Select best checkpoint using validation loss
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Reproducibility
    seed=42,

    # Disable external logging
    report_to="none",
)


# ============================================================
# 14. CREATE TRAINER
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,

    tokenizer=tokenizer,
    data_collator=data_collator,
)


# ============================================================
# 15. CHECK FOR EXISTING CHECKPOINT
# ============================================================

last_checkpoint = None

if os.path.isdir(OUTPUT_DIR):

    last_checkpoint = get_last_checkpoint(
        OUTPUT_DIR
    )


# ============================================================
# 16. START / RESUME TRAINING
# ============================================================

print("\n" + "=" * 60)

if last_checkpoint:

    print("Existing checkpoint found:")
    print(last_checkpoint)
    print("\nResuming training...")

    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

else:

    print("No checkpoint found.")
    print("Starting training from the base IndicTrans2 model...")

    trainer.train()


# ============================================================
# 17. EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("Running validation evaluation...")
print("=" * 60)

evaluation_results = trainer.evaluate()

print("\nEvaluation results:")

for key, value in evaluation_results.items():
    print(f"{key}: {value}")


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("Saving final model...")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

trainer.save_state()

print("\nModel saved to:")
print(OUTPUT_DIR)

print("\nTraining completed successfully!")
print("=" * 60)