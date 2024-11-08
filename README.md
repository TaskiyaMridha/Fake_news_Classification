# Fake News Detection Model

This project uses machine learning techniques to classify news articles as either *real* or *fake* based on text data. Leveraging Natural Language Processing (NLP) techniques and a machine learning classifier, this model helps in identifying misinformation from reliable news sources.

## Overview
The model uses the **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization method to convert text data into numerical features and a **Linear Support Vector Classifier (LinearSVC)** to classify news articles. With an accuracy of approximately 99%, this model provides a solid baseline for detecting fake news.

## Dataset
The dataset consists of two classes:
- **Real**: Legitimate news articles
- **Fake**: Misinformation or unreliable news articles

The data includes:
- **Text**: The news article text.
- **Label**: The classification label (Real/Fake).

### Data Distribution
A bar chart visualizing class distribution is provided to show the balance between real and fake news articles in the dataset.

## Dependencies
To run this project, ensure you have the following libraries installed:
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

Install dependencies using:
```bash
pip install -r requirements.txt
## Model Training and Evaluation

1. TF-IDF Vectorization: Converts the article text into a numerical format suitable for machine learning.
2. Model Training: A LinearSVC model is trained on the vectorized data.
3. Evaluation: The model is evaluated using accuracy and classification metrics, along with a confusion matrix for visual representation.

### Key Code Sections:
```bash
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
### Data Loading and Preprocessing:
```bash
# Load dataset
df = pd.read_csv('/content/fake_and_real_news.csv')

# Split into features (X) and target (Y)
X = df[['Text']]
Y = df['Label']
```
### TF-IDF Transformation:
```bash
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['Text'])
X_test_tfidf = vectorizer.transform(X_test['Text'])
```
### Model Training and Prediction
```bash
clf = LinearSVC(random_state=0)
clf.fit(X_train_tfidf, Y_train)
Y_pred = clf.predict(X_test_tfidf)
```
### Evaluation
```bash
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
print(classification_report(Y_test, Y_pred))
```
### Confusion Matrix:
```bash
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```
## How to Use
1. **Clone this repository:**
```bash
git@github.com:TaskiyaMridha/Fake_news_Classification.git
```
2. **Navigate to the project directory:**
```bash
cd fake_news_detection
```
3. **Install dependancies:**
```bash
pip install -r requirements.txt
```
4. **Run the notebook or script to load the dataset, train the model, and evaluate its performance.**

## Future Improvements

* Experiment with additional models like Random Forest or Naive Bayes.
* Use more complex NLP techniques, like word embeddings, to improve classification accuracy.
* Explore ensemble models for potential accuracy improvements.
