# Efficient-Sentiment-Analysis-of-Twitter-Data-Using-Word2Vec-and-TF-IDF-.
Efficient Sentiment Analysis of Twitter Data Using Word2Vec and TF-IDF with Stratified Cross-Validation
import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_metrics(metrics):
    # Print the metrics
    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F1 Score:", metrics['f1_score'],"\n")


def stratified_k_fold_cv(X, y, model, n_splits=5):
    """
    Perform stratified k-fold cross-validation and return metrics.

    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model: The classifier or regressor model to be evaluated.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        metrics (dict): A dictionary containing mean scores of accuracy, precision, recall, and F1-score.
    """

    skf = StratifiedKFold(n_splits=n_splits)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    metrics = {
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1_score': np.mean(f1_scores)
    }

    return metrics

def calculate_mean_vector(review):
    """
    Calculate the mean vector of a review using Word2Vec embeddings.

    Parameters:
        review (list): List of words in the review.

    Returns:
        mean_vector (numpy array): Mean vector representation of the review.
    """
    vectors = [model.wv[word] for word in review if word in model.wv.index_to_key]
    if len(vectors) == 0:
        return np.zeros(model.vector_size) # Return zero vector if no word vectors found
    return np.mean(vectors, axis=0)

"""**** Word2Vec ****"""

print('Positive records:', len(twitter_samples.strings('positive_tweets.json')))
print('Negative records:', len(twitter_samples.strings('negative_tweets.json')),"\n")

# Load twitter_samples dataset
categories = ['positive_tweets.json', 'negative_tweets.json']

reviews = [(tweet, category)
           for category in categories
           for tweet in twitter_samples.tokenized(category)]


# Initialize and train Word2Vec model
model = Word2Vec([review for review, _ in reviews], min_count=1)


# Create feature vectors and labels
X = [calculate_mean_vector(review) for review, _ in reviews]
y = [category for _, category in reviews]

# Initialize classifiers
nb_clf = GaussianNB()
svm_clf = SVC(kernel='linear')

print("**** Word2Vec ****\n")

# Evaluate Naive Bayes classifier using Word2Vec features
metrics = stratified_k_fold_cv(np.array(X), np.array(y), nb_clf)
print("Naive Bayes Metrics: ")
print_metrics(metrics)

# Evaluate SVM classifier using Word2Vec features
metrics = stratified_k_fold_cv(np.array(X), np.array(y), svm_clf)
print("SVM Metrics: ")
print_metrics(metrics)

"""**** TF-IDF ****"""

# Load twitter_samples dataset
reviews = [(" ".join(tweet), category)
           for category in categories
           for tweet in twitter_samples.tokenized(category)]

# Initialize and fit TF-IDF vectorizer
tweet_strings = [review[0] for review in reviews]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweet_strings)

print("**** TF-IDF ****\n")

# Evaluate Naive Bayes classifier using TF-IDF features
metrics = stratified_k_fold_cv(X.toarray(), np.array(y), nb_clf)
print("Naive Bayes Metrics: ")
print_metrics(metrics)

# Evaluate SVM classifier using TF-IDF features
metrics = stratified_k_fold_cv(X, np.array(y), svm_clf)
print("SVM Metrics: ")
print_metrics(metrics)
