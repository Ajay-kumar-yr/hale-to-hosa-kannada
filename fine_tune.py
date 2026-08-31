
import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_datasets",
    "train_final.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_datasets",
    "validation_final.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "hale_hosa_model"
)

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128


# ============================================================
# 2. TEST MODE
# ============================================================

# IMPORTANT:
# True  = run a small test before full training
# False = train the complete dataset

TEST_MODE = True

TEST_TRAIN_SIZE = 100
TEST_VALIDATION_SIZE = 20


# ============================================================
# 3. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("HALE KANNADA → HOSA KANNADA FINE-TUNING")
print("=" * 70)

print("PyTorch :", torch.__version__)
print("Device  :", device)


if torch.cuda.is_available():

    print("GPU     :", torch.cuda.get_device_name(0))
    print("CUDA    :", torch.version.cuda)

else:

    print()
    print("WARNING: CUDA is not available.")
    print("Training will run on CPU.")
    print("CPU training can be very slow.")


# ============================================================
# 4. CHECK DATASET FILES
# ============================================================

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_FILE}"
    )


if not os.path.exists(VALIDATION_FILE):

    raise FileNotFoundError(
        f"\nValidation file not found:\n{VALIDATION_FILE}"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)

validation_df = pd.read_csv(VALIDATION_FILE)


print("Original train shape      :", train_df.shape)
print("Original validation shape :", validation_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "source",
    "target"
]

for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"\nColumn '{column}' not found in train.csv.\n"
            f"Available columns: {train_df.columns.tolist()}"
        )

    if column not in validation_df.columns:

        raise ValueError(
            f"\nColumn '{column}' not found in validation.csv.\n"
            f"Available columns: {validation_df.columns.tolist()}"
        )


# ============================================================
# 7. SELECT ONLY SOURCE AND TARGET
# ============================================================

train_df = train_df[
    ["source", "target"]
].copy()

validation_df = validation_df[
    ["source", "target"]
].copy()


# ============================================================
# 8. REMOVE MISSING VALUES
# ============================================================

train_df = train_df.dropna(
    subset=["source", "target"]
)

validation_df = validation_df.dropna(
    subset=["source", "target"]
)


# ============================================================
# 9. CONVERT TO STRING
# ============================================================

train_df["source"] = (
    train_df["source"]
    .astype(str)
    .str.strip()
)

train_df["target"] = (
    train_df["target"]
    .astype(str)
    .str.strip()
)

validation_df["source"] = (
    validation_df["source"]
    .astype(str)
    .str.strip()
)

validation_df["target"] = (
    validation_df["target"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 10. REMOVE EMPTY ROWS
# ============================================================

train_df = train_df[
    (train_df["source"] != "") &
    (train_df["target"] != "")
].copy()

validation_df = validation_df[
    (validation_df["source"] != "") &
    (validation_df["target"] != "")
].copy()


print("\nAfter cleaning:")
print("Train      :", train_df.shape)
print("Validation :", validation_df.shape)


# ============================================================
# 11. SHOW SAMPLE
# ============================================================

print("\nSample training pair:")
print("-" * 70)

print("Hale Kannada :")
print(train_df.iloc[0]["source"])

print("\nHosa Kannada :")
print(train_df.iloc[0]["target"])

print("-" * 70)


# ============================================================
# 12. SMALL TEST DATASET
# ============================================================

if TEST_MODE:

    print("\n" + "=" * 70)
    print("TEST MODE ENABLED")
    print("=" * 70)

    print(
        f"Using only {TEST_TRAIN_SIZE} training examples"
    )

    print(
        f"Using only {TEST_VALIDATION_SIZE} validation examples"
    )

    train_df = train_df.head(
        TEST_TRAIN_SIZE
    ).copy()

    validation_df = validation_df.head(
        TEST_VALIDATION_SIZE
    ).copy()


print("\nFinal dataset used for this run:")

print("Train      :", len(train_df))
print("Validation :", len(validation_df))


# ============================================================
# 13. CONVERT TO HUGGING FACE DATASETS
# ============================================================

print("\nConverting datasets...")

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df,
    preserve_index=False
)


print("✅ Dataset conversion completed")


# ============================================================
# 14. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("✅ Tokenizer loaded")


# ============================================================
# 15. LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)

print("✅ Model loaded")


# ============================================================
# 16. PREPROCESSING
# ============================================================

def preprocess_function(examples):

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Do NOT add:
    #
    # kan_Knda kan_Knda
    #
    # to the input.
    #
    # The model learns the Hale Kannada → Hosa Kannada
    # transformation from the parallel training pairs.
    #

    source_texts = [
        f"kan_Knda kan_Knda {text}"
        for text in examples["source"]
    ]
    target_texts = examples["target"]


    # --------------------------------------------------------
    # TOKENIZE SOURCE
    # --------------------------------------------------------

    model_inputs = tokenizer(
        source_texts,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True
    )


    # --------------------------------------------------------
    # TOKENIZE TARGET
    # --------------------------------------------------------

    labels = tokenizer(
        text_target=target_texts,
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )


    # --------------------------------------------------------
    # ADD LABELS
    # --------------------------------------------------------

    model_inputs["labels"] = labels["input_ids"]


    return model_inputs


# ============================================================
# 17. TOKENIZE TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING TRAINING DATA")
print("=" * 70)

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

print("✅ Training data tokenized")


# ============================================================
# 18. TOKENIZE VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING VALIDATION DATA")
print("=" * 70)

tokenized_validation = validation_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=validation_dataset.column_names
)

print("✅ Validation data tokenized")


# ============================================================
# 19. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

print("\n✅ Data collator created")


# ============================================================
# 20. TRAINING SETTINGS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)


training_args = Seq2SeqTrainingArguments(

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir=OUTPUT_DIR,

    overwrite_output_dir=True,


    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------

    num_train_epochs=1 if TEST_MODE else 3,


    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,


    # --------------------------------------------------------
    # GRADIENT ACCUMULATION
    # --------------------------------------------------------

    gradient_accumulation_steps=8,


    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    learning_rate=5e-5,

    weight_decay=0.01,

    warmup_ratio=0.1,


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=50 if TEST_MODE else 500,


    # --------------------------------------------------------
    # SAVING
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=50 if TEST_MODE else 500,

    save_total_limit=2,


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=10 if TEST_MODE else 50,


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    predict_with_generate=True,


    # --------------------------------------------------------
    # GRADIENT CHECKPOINTING
    # --------------------------------------------------------

    gradient_checkpointing=True,


    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    fp16=False,


    # --------------------------------------------------------
    # REPORTING
    # --------------------------------------------------------

    report_to="none"
)


# ============================================================
# 21. PRINT CONFIGURATION
# ============================================================

print("Epochs                  :", training_args.num_train_epochs)
print("Train batch size       :", training_args.per_device_train_batch_size)
print("Validation batch size  :", training_args.per_device_eval_batch_size)
print("Gradient accumulation  :", training_args.gradient_accumulation_steps)
print("Learning rate           :", training_args.learning_rate)
print("Output directory        :", OUTPUT_DIR)


# ============================================================
# 22. CREATE TRAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAINER")
print("=" * 70)


trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    processing_class=tokenizer,

    data_collator=data_collator
)


print("✅ Trainer created")


# ============================================================
# 23. START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

print("\nTask:")
print("Hale Kannada → Hosa Kannada")

print("\nTraining examples   :", len(tokenized_train))
print("Validation examples :", len(tokenized_validation))

if TEST_MODE:

    print("\n⚠️ TEST MODE")
    print("Only a small dataset will be trained.")

else:

    print("\n⚠️ FULL TRAINING MODE")
    print("Full dataset will be trained.")


if device == "cpu":

    print("\n⚠️ WARNING")
    print("CPU training can be VERY SLOW.")


# ============================================================
# 24. TRAIN
# ============================================================

train_result = trainer.train()


# ============================================================
# 25. TRAINING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print("\nTraining result:")
print(train_result)


# ============================================================
# 26. EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

evaluation_result = trainer.evaluate()

print("\nEvaluation result:")
print(evaluation_result)


# ============================================================
# 27. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# 28. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETED")
print("=" * 70)

print("\nModel saved at:")

print(OUTPUT_DIR)

print("\nTEST_MODE:", TEST_MODE)

if TEST_MODE:

    print("\nThis was only a TEST RUN.")

    print(
        "If everything worked correctly, change:"
    )

    print(
        "TEST_MODE = False"
    )

    print(
        "and run train.py again for full training."
    )

else:

    print(
        "\nFull Hale Kannada → Hosa Kannada training completed."
    )
