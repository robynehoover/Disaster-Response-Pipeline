import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import sqlalchemy as sal
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    '''
    Function that imports a SQLite databse and outputs variables for machine learning pipeline
    Input: database filepath
    Output: X,y,category_names variables
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    
    # Child alone contains only zeros and should be removed
    df = df.drop(['child_alone'],axis=1)
    
    X = df.message
    y = df.iloc[:,4:]
    category_names = list(np.array(y.columns))
    
    return X, y, category_names

def tokenize(text):
    '''
    Function that tokenizes raw text
    Import: text
    Output: clean tokens
    '''
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    Builds a Multioutput classifier pipeline
    Input: none
    Output: Multi Output classifier pipeline
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters_grid = {'classifier__estimator__learning_rate': [2, 3, 4],
              'classifier__estimator__n_estimators': [50, 100, 150]}

    model = GridSearchCV(pipeline, param_grid=parameters_grid, n_jobs=4, verbose=2)
    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Input: model, X_test, y_test, category_names
    Output: classification report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
          
    
def save_model(model, model_filepath):
    '''
    Saves model to a pickle file
    '''
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
