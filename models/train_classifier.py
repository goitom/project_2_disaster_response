# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

def load_data(database_filepath):
    """
    Loads table from database as a Pandas Dataframe
    and returns the following:
    X -- feature dataset containing the messages to be 
         categorized
    y -- label dataset containing the 36 categories that each
         message is assigned to.
    category_names -- list containing category names

    Keyword arguments:
    database_filepath -- filepath (including file name) 
                         of the database containing the 
                         messages and categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_and_categories', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(y.columns)

    return X, y, category_names

def tokenize(text):
    """
    Cleans, tokenizes, lemmatizes messages in preparation for 
    classification algorithm

    1) finds and replaces urls with a placeholder
    2) finds and replaces non alphanumeric characters with a space
    3) removes stop words from tokenized messages
    4) strips leading and trailing spaces and lowcases lemmatized 
       tokens

    Keyword arguments:
    text -- raw message that will be cleaned, tokenized 
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    text = re.sub(r'\W+', ' ', text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Creates a pipeline and grid search for hyperparameter tuning
    returns pipeline with the specified parameter search space
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2),(2,2)),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__estimator__learning_rate': [0.1, 0.5],
        'clf__estimator__n_estimators': [50, 60, 70]
        }
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=216)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Generates predicted values for test data based on
    fit model. Outputs a classification report for each category.

    Keyword arguments:
    model -- fit model based on training data
    X_test, Y_test -- message and target category values for testing
    category_names -- list of possible categories for each message 
    """
    Y_pred = model.predict(X_test)
    for i, label in enumerate(category_names):
        print(category_names[i])
        print(classification_report(Y_test[label], Y_pred[:,i]))



def save_model(model, model_filepath):
    """
    Export the classifier to a pickle file

    Keyword arguments:
    model -- final model
    model_filepath -- location and name of saved pickle file
    """
    with open(model_filepath, 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_score_)
        print(model.best_params_)

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