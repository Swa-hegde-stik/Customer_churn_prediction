# Customer Churn Prediction

## Project Overview

This project builds a **customer churn prediction model** using a banking customer dataset. The objective is to predict whether a customer will leave the bank (`Exited = 1`) based on customer and account-related features.

## Dataset

- **Records:** 10,000 customers
- **Target:** `Exited`
- **Features include:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Credit Card status, Active Member status, and Estimated Salary.
- **Missing values:** None

## Workflow

### 1. Data Loading & Exploration

The dataset was loaded using Pandas and examined using:

```python
df.head()
df.info()
df.isnull().sum()
```

### 2. Data Preprocessing

- Applied **one-hot encoding** to `Geography` and `Gender`.
- Removed the `Surname` column.
- Separated features (`X`) and target (`y`).
- Applied **StandardScaler** for feature scaling.
- Split the data into **80% training** and **20% testing** sets.

```python
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int
)

df = df.drop(["Surname"], axis=1)
```

### 3. Neural Network Model

A **Multi-Layer Perceptron (MLP)** was built using TensorFlow/Keras.

Architecture:

```text
Input Layer
     ↓
Dense Layer (3 neurons, Sigmoid)
     ↓
Output Layer (1 neuron, Sigmoid)
```

The model was trained using:

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Epochs:** 20
- **Task:** Binary Classification

```python
model.compile(
    optimizer="Adam",
    loss="binary_crossentropy"
)

model.fit(x_train, y_train, epochs=20)
```

### 4. Prediction & Evaluation

The model outputs a probability between 0 and 1. A threshold of **0.5** was used to convert probabilities into binary predictions.

```python
y_log = model.predict(x_test)
y_pred = np.where(y_log > 0.5, 1, 0)
```

### Result

The model achieved **82.6% test accuracy** on the held-out test set.

```python
accuracy_score(y_test, y_pred)
# 0.826
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## Key Concepts Demonstrated

- Data preprocessing
- Categorical feature encoding
- Feature scaling
- Train-test splitting
- Binary classification
- Neural networks / MLP
- Sigmoid activation
- Binary cross-entropy loss
- Model evaluation using accuracy

## Project Structure

```text
Customer-Churn-Prediction/
│
├── Churn_prediction.ipynb
├── Churn_Modelling.csv
└── README.md
```

## Conclusion

The project demonstrates an end-to-end machine learning workflow for customer churn prediction, from data preprocessing and feature preparation to neural network training and evaluation.
