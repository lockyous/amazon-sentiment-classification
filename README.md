# Amazon Sentiment Classification

This project analyzes Amazon product reviews and builds machine learning models to classify customer sentiment as **positive or negative** based on review text.

Understanding sentiment in customer feedback helps businesses identify product strengths, detect issues earlier, and improve overall customer experience.

---

## Project Overview

Online reviews strongly influence purchasing decisions. With hundreds of thousands of reviews available, manually analyzing customer opinions becomes impractical.

The goal of this project is to:

- preprocess large-scale review text data
- transform text into numerical features
- train classification models
- evaluate sentiment prediction performance

The final objective is to develop a model capable of automatically determining whether a review expresses **positive or negative sentiment**.

---

## Dataset

**Dataset:** Amazon Product Reviews  

**Source:**  
https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews

**Size**

```
568,454 reviews
10 columns
```

**Target Variable**

```
Sentiment
1 = Positive (Score 4–5)
0 = Negative (Score 1–2)
```

Reviews with a score of **3 (neutral)** were removed to simplify the task into a binary classification problem.

⚠️ The dataset is **not included in the repository due to its large size**.

---

## Project Pipeline

### 1. Data Exploration

Initial exploration included:

- dataset structure inspection
- missing value analysis
- summary statistics
- review text length analysis

### 2. Feature Engineering

The dataset was cleaned by:

- removing identifier columns (`Id`, `UserId`, `ProductId`)
- removing metadata columns
- dropping rows with missing text
- converting review scores into binary sentiment labels

### 3. Exploratory Data Analysis

EDA was performed to understand the structure of the review text.

Analysis included:

- sentiment class distribution
- vocabulary size comparison
- word count comparison between `Summary` and `Text`

Results showed that **Summary contained limited additional information compared to the full review text**, so it was removed before modeling.

### 4. Text Preprocessing

Text data was cleaned using the following steps:

- lowercase conversion
- HTML tag removal
- URL removal
- punctuation filtering
- stopword removal

This produced a cleaner text representation suitable for machine learning models.

### 5. Feature Vectorization

Text data was converted into numerical features using:

```
TF-IDF Vectorization
```

TF-IDF measures the importance of each word within the review corpus.

### 6. Train-Test Split

The dataset was split into training and testing sets.

```
70% Training Data
30% Testing Data
```

Stratified sampling preserved the original sentiment distribution.

---

## Models

Two classification models were trained and evaluated.

### Multinomial Naive Bayes

A probabilistic model widely used for text classification.

Hyperparameter tuning was applied using **GridSearchCV** to improve performance on the minority (negative) class.

**Performance after tuning**

```
Negative F1-score: 0.71
Positive F1-score: 0.94
Weighted F1-score: 0.90
```

### Logistic Regression

A linear classifier that models the probability of positive or negative sentiment.

After hyperparameter tuning, Logistic Regression delivered the strongest performance.

**Performance after tuning**

```
Negative F1-score: 0.80
Positive F1-score: 0.97
Weighted F1-score: 0.94
```

---

## Model Comparison

| Model | Negative F1 | Positive F1 | Weighted F1 |
|------|-------------|-------------|-------------|
| Naive Bayes | 0.71 | 0.94 | 0.90 |
| Logistic Regression | **0.80** | **0.97** | **0.94** |

Logistic Regression showed the most **balanced and reliable performance** across both classes and was selected as the final model.

---

## Conclusion

This project demonstrates how large-scale text data can be transformed into actionable insights using natural language processing and machine learning techniques.

Key takeaways:

- Text preprocessing significantly improves model performance
- TF-IDF effectively converts text into numerical features
- Logistic Regression provided the most reliable sentiment classification performance

Automated sentiment classification enables companies to analyze customer feedback at scale and make better product and business decisions.

---

## Project Structure

```
amazon-sentiment-classification
│
├── sentiment_analysis.ipynb
├── sentiment_analysis.py
├── sentiment_analysis.html
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Author

**Jeongwoo Kim**  
Arizona State University  
Data Science
