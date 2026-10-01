# ============================================================
# AI & ML INTERNSHIP - TASK 1
# Data Cleaning & Preprocessing
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

# Path to the input dataset
DATA_PATH = "Titanic-Dataset.csv"

# Path for the cleaned dataset
OUTPUT_PATH = "processed_titanic.csv"

# Folder for saving visualizations
PLOT_FOLDER = "plots"

# Create plots folder if it does not exist
os.makedirs(PLOT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("=" * 60)
print("AI & ML INTERNSHIP - TASK 1")
print("DATA CLEANING & PREPROCESSING")
print("=" * 60)

print("\n[1] Loading dataset...")

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\nERROR: Dataset not found at: {DATA_PATH}")
    print("Make sure your Titanic CSV is inside the 'data' folder.")
    exit()

print("Dataset loaded successfully!")


# ------------------------------------------------------------
# 4. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. BASIC DATASET INFORMATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))


# ------------------------------------------------------------
# 5. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. CHECKING DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")


# ------------------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. CHECKING MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing values in each column:")
print(missing_values)

print("\nMissing-value percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))


# ------------------------------------------------------------
# 7. VISUALIZE MISSING VALUES
# ------------------------------------------------------------

print("\nCreating missing-value heatmap...")

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values Before Cleaning")
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_FOLDER, "missing_values_before.png")
)

plt.show()


# ------------------------------------------------------------
# 8. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. HANDLING MISSING VALUES")
print("=" * 60)

# Age - numerical column
if "Age" in df.columns:
    age_missing = df["Age"].isnull().sum()

    if age_missing > 0:
        age_median = df["Age"].median()

        df["Age"] = df["Age"].fillna(age_median)

        print(
            f"Age: Filled {age_missing} missing values "
            f"using median = {age_median:.2f}"
        )
    else:
        print("Age: No missing values.")

# Embarked - categorical column
if "Embarked" in df.columns:
    embarked_missing = df["Embarked"].isnull().sum()

    if embarked_missing > 0:
        embarked_mode = df["Embarked"].mode()[0]

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print(
            f"Embarked: Filled {embarked_missing} missing values "
            f"using mode = '{embarked_mode}'"
        )
    else:
        print("Embarked: No missing values.")

# Cabin - categorical column
if "Cabin" in df.columns:
    cabin_missing = df["Cabin"].isnull().sum()

    if cabin_missing > 0:
        df["Cabin"] = df["Cabin"].fillna("Unknown")

        print(
            f"Cabin: Filled {cabin_missing} missing values "
            "with 'Unknown'"
        )
    else:
        print("Cabin: No missing values.")


# ------------------------------------------------------------
# 9. CHECK MISSING VALUES AFTER CLEANING
# ------------------------------------------------------------

print("\nMissing values after cleaning:")

remaining_missing = df.isnull().sum()

print(remaining_missing)

if remaining_missing.sum() == 0:
    print("\nAll missing values have been handled.")
else:
    print("\nSome missing values still remain.")


# ------------------------------------------------------------
# 10. REMOVE UNNECESSARY COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. REMOVING UNNECESSARY COLUMNS")
print("=" * 60)

# These columns are identifiers/text fields and are not required
# for this basic preprocessing task.

columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

# Only remove columns that actually exist
columns_to_drop = [
    column for column in columns_to_drop
    if column in df.columns
]

if columns_to_drop:
    df = df.drop(columns=columns_to_drop)

    print("Removed columns:")
    print(columns_to_drop)
else:
    print("No unnecessary columns found.")


# ------------------------------------------------------------
# 11. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

# Encode Sex using label/binary encoding
if "Sex" in df.columns:

    print("\nEncoding 'Sex':")

    print("Before:")
    print(df["Sex"].unique())

    df["Sex"] = df["Sex"].map({
        "male": 0,
        "female": 1
    })

    print("After:")
    print(df["Sex"].unique())


# One-hot encode Embarked
if "Embarked" in df.columns:

    print("\nOne-hot encoding 'Embarked'...")

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    print("Embarked successfully encoded.")


# ------------------------------------------------------------
# 12. DISPLAY DATA AFTER ENCODING
# ------------------------------------------------------------

print("\nDataset after encoding:")
print(df.head())

print("\nData types after encoding:")
print(df.dtypes)


# ------------------------------------------------------------
# 13. DETECT OUTLIERS USING BOXPLOTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. OUTLIER DETECTION")
print("=" * 60)

# Numerical columns we want to investigate
outlier_columns = [
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

# Keep only columns that exist
outlier_columns = [
    column for column in outlier_columns
    if column in df.columns
]

for column in outlier_columns:

    print(f"\nCreating boxplot for {column}...")

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_FOLDER,
            f"boxplot_{column}.png"
        )
    )

    plt.show()


# ------------------------------------------------------------
# 14. REMOVE OUTLIERS USING IQR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("9. REMOVING OUTLIERS USING IQR")
print("=" * 60)

# We will remove extreme values from Fare.
#
# IQR = Q3 - Q1
#
# Lower bound = Q1 - 1.5 * IQR
# Upper bound = Q3 + 1.5 * IQR

if "Fare" in df.columns:

    Q1 = df["Fare"].quantile(0.25)
    Q3 = df["Fare"].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")

    rows_before = len(df)

    df = df[
        (df["Fare"] >= lower_bound) &
        (df["Fare"] <= upper_bound)
    ]

    rows_after = len(df)

    rows_removed = rows_before - rows_after

    print(f"\nRows before outlier removal: {rows_before}")
    print(f"Rows after outlier removal: {rows_after}")
    print(f"Outliers removed: {rows_removed}")


# ------------------------------------------------------------
# 15. BOXPLLOT AFTER OUTLIER REMOVAL
# ------------------------------------------------------------

if "Fare" in df.columns:

    print("\nCreating boxplot after outlier removal...")

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df["Fare"])

    plt.title("Fare After Outlier Removal")
    plt.xlabel("Fare")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_FOLDER,
            "fare_after_outlier_removal.png"
        )
    )

    plt.show()


# ------------------------------------------------------------
# 16. STANDARDIZE NUMERICAL FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("10. STANDARDIZING NUMERICAL FEATURES")
print("=" * 60)

# Numerical features to standardize
numeric_columns = [
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

numeric_columns = [
    column for column in numeric_columns
    if column in df.columns
]

print("\nColumns being standardized:")
print(numeric_columns)

if numeric_columns:

    scaler = StandardScaler()

    df[numeric_columns] = scaler.fit_transform(
        df[numeric_columns]
    )

    print("\nStandardization completed.")

    print("\nStandardized numerical features:")
    print(df[numeric_columns].head())


# ------------------------------------------------------------
# 17. FINAL DATASET CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("11. FINAL DATASET CHECK")
print("=" * 60)

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal data types:")
print(df.dtypes)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nFinal dataset preview:")
print(df.head())


# ------------------------------------------------------------
# 18. FINAL STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\nFinal statistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 19. SAVE PROCESSED DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("12. SAVING PROCESSED DATASET")
print("=" * 60)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nProcessed dataset saved successfully:")
print(OUTPUT_PATH)


# ------------------------------------------------------------
# 20. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nCompleted preprocessing steps:")
print("1. Loaded dataset")
print("2. Explored dataset")
print("3. Checked missing values")
print("4. Handled missing values")
print("5. Removed unnecessary columns")
print("6. Encoded categorical variables")
print("7. Visualized outliers")
print("8. Removed outliers using IQR")
print("9. Standardized numerical features")
print("10. Saved processed dataset")

print("\nOutput files:")
print(f"- {OUTPUT_PATH}")
print(f"- {PLOT_FOLDER}/")