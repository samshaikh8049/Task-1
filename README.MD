# AI & ML Internship - Task 1

## Data Cleaning & Preprocessing

### 📌 Project Overview

This project is completed as part of the **AI & ML Internship - Task 1: Data Cleaning & Preprocessing**.

The objective of this task is to learn how to clean and prepare a raw dataset for use in Machine Learning.

For this task, the **Titanic dataset** is used.

The preprocessing workflow includes:

* Loading and exploring the dataset
* Checking data types and missing values
* Handling missing values
* Removing duplicate records
* Removing unnecessary columns
* Encoding categorical variables
* Detecting and visualizing outliers
* Removing outliers using the IQR method
* Standardizing numerical features
* Saving the processed dataset

---

## 🎯 Objective

The main objective is to transform raw data into a clean and machine-learning-ready dataset.

The task focuses on the following preprocessing techniques:

1. Data exploration
2. Missing-value handling
3. Categorical encoding
4. Outlier detection and removal
5. Numerical feature standardization

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

## 📂 Project Structure


```text
Task-1/
│
├── README.MD
├── Titanic-Dataset.csv
├── task1.py
├── processed_titanic.csv
│
├── missing_values_before.png
├── boxplot_Age.png
├── boxplot_SibSp.png
├── boxplot_Parch.png
├── boxplot_Fare.png
└── fare_after_outlier_removal.png

---

## 📊 Dataset

### Titanic Dataset

The Titanic dataset contains information about passengers aboard the Titanic.

Some of the important columns include:

* `PassengerId` - Passenger identification number
* `Survived` - Target variable indicating survival
* `Pclass` - Passenger class
* `Name` - Passenger name
* `Sex` - Passenger gender
* `Age` - Passenger age
* `SibSp` - Number of siblings/spouses aboard
* `Parch` - Number of parents/children aboard
* `Ticket` - Ticket number
* `Fare` - Passenger fare
* `Cabin` - Cabin information
* `Embarked` - Port of embarkation

---

# 🔄 Data Preprocessing Steps

## 1. Load the Dataset

The Titanic CSV file is loaded using Pandas.

```python
df = pd.read_csv("data/titanic.csv")
```

Basic information about the dataset is then examined using:

```python
df.head()
df.shape
df.info()
df.describe()
```

---

## 2. Check Duplicate Rows

Duplicate rows are checked using:

```python
df.duplicated().sum()
```

If duplicates are present, they are removed using:

```python
df.drop_duplicates()
```

---

## 3. Check Missing Values

Missing values are identified using:

```python
df.isnull().sum()
```

The percentage of missing values in each column is also calculated.

A heatmap is created using Seaborn to visualize missing values.

The generated visualization is saved inside the `plots` directory.

---

## 4. Handle Missing Values

Different techniques are used depending on the type of column.

### Age

`Age` is a numerical feature, so missing values are filled using the median:

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

### Embarked

`Embarked` is categorical, so missing values are filled using the mode:

```python
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)
```

### Cabin

Missing cabin values are represented as:

```text
Unknown
```

This prevents missing values from remaining in the dataset before further preprocessing.

---

## 5. Remove Unnecessary Columns

The following columns are removed for this preprocessing task:

```text
PassengerId
Name
Ticket
Cabin
```

These columns are not required for the basic preprocessing workflow implemented in this project.

---

## 6. Encode Categorical Variables

Machine Learning algorithms generally require numerical input.

### Sex

The `Sex` column is converted into numerical values:

```text
male   → 0
female → 1
```

### Embarked

One-hot encoding is applied to the `Embarked` column using:

```python
pd.get_dummies()
```

This converts categorical values into numerical columns.

---

## 7. Detect Outliers

Boxplots are used to visualize possible outliers in numerical features.

The following columns are examined:

* `Age`
* `SibSp`
* `Parch`
* `Fare`

Example:

```python
sns.boxplot(x=df["Fare"])
plt.show()
```

The generated boxplots are saved inside the `plots` directory.

---

## 8. Remove Outliers

The **Interquartile Range (IQR)** method is used to identify extreme values in the `Fare` column.

The calculation is:

```text
IQR = Q3 - Q1
```

The lower and upper limits are calculated as:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Rows outside these limits are removed.

The number of rows before and after outlier removal is displayed by the Python script.

---

## 9. Standardize Numerical Features

The following numerical features are standardized:

* `Age`
* `SibSp`
* `Parch`
* `Fare`

`StandardScaler` from Scikit-learn is used.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[numeric_columns] = scaler.fit_transform(
    df[numeric_columns]
)
```

Standardization transforms the numerical features so that they have approximately:

```text
Mean = 0
Standard Deviation = 1
```

---

## 10. Save the Processed Dataset

After preprocessing is completed, the cleaned dataset is saved as:

```text
processed_titanic.csv
```

The output file is generated using:

```python
df.to_csv("processed_titanic.csv", index=False)
```

---

# 📈 Visualizations

The project generates visualizations to help understand the dataset.

The following plots are created:

### Missing Values

```text
missing_values_before.png
```

This shows where missing values occur in the original dataset.

### Boxplots

```text
boxplot_Age.png
boxplot_SibSp.png
boxplot_Parch.png
boxplot_Fare.png
```

These plots are used to inspect potential outliers.

### Fare After Outlier Removal

```text
fare_after_outlier_removal.png
```

This allows the distribution of `Fare` to be inspected after the IQR-based outlier removal.

---

# ▶️ How to Run the Project

## Step 1: Install Python

Make sure Python is installed on your computer.

Check the installation using:

```bash
python --version
```

---

## Step 2: Install Required Libraries

Run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Step 3: Place the Dataset

Place the Titanic CSV file inside:

```text
data/titanic.csv
```

The final structure should look like:

```text
task-1-data-cleaning/
│
├── data/
│   └── titanic.csv
│
└── task1_preprocessing.py
```

---

## Step 4: Run the Python Script

Open a terminal inside the project directory and run:

```bash
python task1_preprocessing.py
```

---

## Step 5: Check the Output

After successful execution, the project will generate:

```text
processed_titanic.csv
```

and the visualization files inside:

```text
plots/
```

---

# ✅ Task Completion Summary

The following preprocessing tasks were completed:

| Task                      | Status      |
| ------------------------- | ----------- |
| Dataset loading           | ✅ Completed |
| Dataset exploration       | ✅ Completed |
| Data type inspection      | ✅ Completed |
| Missing-value detection   | ✅ Completed |
| Missing-value handling    | ✅ Completed |
| Duplicate detection       | ✅ Completed |
| Duplicate removal         | ✅ Completed |
| Categorical encoding      | ✅ Completed |
| Outlier visualization     | ✅ Completed |
| Outlier removal           | ✅ Completed |
| Numerical standardization | ✅ Completed |
| Processed dataset export  | ✅ Completed |

---

# 🎓 Key Concepts Learned

Through this task, the following Machine Learning preprocessing concepts were practiced:

### Missing Values

Missing values are values that are not present in a dataset. They can be handled using methods such as mean, median, mode, or other imputation techniques.

### Categorical Encoding

Categorical variables need to be converted into numerical representations before being supplied to many Machine Learning algorithms.

### Standardization

Standardization transforms numerical features so that they have approximately zero mean and unit variance.

### Outliers

Outliers are observations that are unusually far from the rest of the data. Boxplots and the IQR method can be used to identify them.

### Data Preprocessing

Data preprocessing is important because raw datasets often contain missing values, categorical variables, inconsistent information, and outliers that need to be handled before Machine Learning algorithms are applied.

---

# 📁 Files Included

### `task1_preprocessing.py`

Python script containing the complete data cleaning and preprocessing workflow.

### `data/titanic.csv`

Original Titanic dataset used for the task.

### `processed_titanic.csv`

Dataset generated after preprocessing.

### `plots/`

Contains visualizations generated during data cleaning and outlier detection.

### `README.md`

Documentation explaining the project, preprocessing steps, and how to run the code.

---

# 📌 Conclusion

The Titanic dataset was successfully cleaned and preprocessed for Machine Learning.

The project demonstrates the complete basic preprocessing workflow, including:

**Data Exploration → Missing Value Handling → Encoding → Outlier Detection → Outlier Removal → Standardization → Processed Dataset**

This provides a clean dataset that can be used as input for further Machine Learning tasks.

---

## Internship Task

**AI & ML Internship - Task 1: Data Cleaning & Preprocessing**

Tools used:

**Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn**
