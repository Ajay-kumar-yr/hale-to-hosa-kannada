import pandas as pd
import re
import os


# --------------------------------------------------
# 1. Load original datasets
# --------------------------------------------------
base_path = "sft_training_silver_high_precision"

train = pd.read_csv(
    f"{base_path}/train.csv",
    low_memory=False
)

validation = pd.read_csv(
    f"{base_path}/validation.csv",
    low_memory=False
)

test = pd.read_csv(
    f"{base_path}/test.csv",
    low_memory=False
)


# --------------------------------------------------
# 2. Keep only useful translation tasks
# --------------------------------------------------

selected_tasks = [
    "modernize_word",
    "gloss_poem_segment"
]

train = train[train["task_type"].isin(selected_tasks)].copy()
validation = validation[validation["task_type"].isin(selected_tasks)].copy()
test = test[test["task_type"].isin(selected_tasks)].copy()


# --------------------------------------------------
# 3. Extract actual Halegannada text from prompt
# --------------------------------------------------

def extract_source(row):

    prompt = str(row["prompt"])
    task = row["task_type"]

    # ----------------------------------------------
    # modernize_word
    # ----------------------------------------------
    if task == "modernize_word":

        # Everything after "ಪದ:"
        match = re.search(r"ಪದ:\s*(.*)", prompt)

        if match:
            return match.group(1).strip()

        return ""


    # ----------------------------------------------
    # gloss_poem_segment
    # ----------------------------------------------
    elif task == "gloss_poem_segment":

        # Everything after "ಸಾಲು:"
        match = re.search(r"ಸಾಲು:\s*(.*)", prompt, re.DOTALL)

        if match:
            return match.group(1).strip()

        return ""


    return ""


# --------------------------------------------------
# 4. Apply extraction
# --------------------------------------------------

train["source"] = train.apply(extract_source, axis=1)
validation["source"] = validation.apply(extract_source, axis=1)
test["source"] = test.apply(extract_source, axis=1)


# --------------------------------------------------
# 5. Target = completion
# --------------------------------------------------

train["target"] = train["completion"].astype(str).str.strip()
validation["target"] = validation["completion"].astype(str).str.strip()
test["target"] = test["completion"].astype(str).str.strip()


# --------------------------------------------------
# 6. Keep only source and target
# --------------------------------------------------

train = train[["source", "target"]]
validation = validation[["source", "target"]]
test = test[["source", "target"]]


# --------------------------------------------------
# 7. Remove empty rows
# --------------------------------------------------

train = train[
    (train["source"].str.len() > 0) &
    (train["target"].str.len() > 0)
]

validation = validation[
    (validation["source"].str.len() > 0) &
    (validation["target"].str.len() > 0)
]

test = test[
    (test["source"].str.len() > 0) &
    (test["target"].str.len() > 0)
]


# --------------------------------------------------
# 8. Create output folder
# --------------------------------------------------

output_folder = "cleaned_datasets"

os.makedirs(output_folder, exist_ok=True)


# --------------------------------------------------
# 9. Save final datasets
# --------------------------------------------------

train.to_csv(
    f"{output_folder}/train_final.csv",
    index=False,
    encoding="utf-8-sig"
)

validation.to_csv(
    f"{output_folder}/validation_final.csv",
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    f"{output_folder}/test_final.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 10. Display information
# --------------------------------------------------

print("\n====================================")
print("DATA PREPARATION COMPLETED")
print("====================================")

print("\nTrain:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

print("\nExample:")
print("------------------------------------")

print("SOURCE:")
print(train.iloc[0]["source"])

print("\nTARGET:")
print(train.iloc[0]["target"])

print("\nFiles saved in:")
print(output_folder)

print("\n====================================")