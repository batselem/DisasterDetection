import sys
# import libraries
import re
import matplotlib.pyplot as plt
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine


def load_data(database_filepath):

    """ Load data from database into dataframe.

    Args:
        database_filepath: String.
    Returns:
       X: numpy.ndarray. Disaster messages.
       Y: numpy.ndarray. Disaster categories for each messages.
       category_name: list. Disaster category names.
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_category', con=engine)
    
    categories = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[categories].values

    return X, y, categories.values

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):

    """Tokenize text.
    Args:
        text: String. A disaster message.
    Returns:
        list. It contains tokens.
    """

    # Parse urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalize and tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()

    # Lemmatize
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Get results and add them to a dataframe.
def display_result(y_test, y_pred, category_names):

    """ Load data from database into dataframe.

    Args:
        y_test: numpy.ndarray. True disaster categories.
        y_pred: numpy.ndarray. Predicted disaster categories.
       category_name: list. Disaster category names.

    Returns:
       results: Dataframe. F-score, precision and recall for each prediction.
    """

    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])

    for num, cat in enumerate(category_names):
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[:,num], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)

    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    return results

def build_model():

    """Build model.
    Returns: 
        pipline: sklearn.model_selection.GridSearchCV. Random Forest Classifier.
    """

    # Set machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Set parameters for gird search
    parameters = {'clf__estimator__n_estimators': [20, 50]}

    # Set grid search to find optimal parameters.
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model
    Args:
        model: sklearn.model_selection.GridSearchCV. 
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    Y_pred = model.predict(X_test)
    display_result(Y_test, Y_pred, category_names)

def save_model(model, model_filepath):
    """Save model
    Args:
        model: sklearn.model_selection.GridSearchCV.
        model_filepath: String. Save the trained model as a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()