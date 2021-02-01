#import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords') 
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import hamming_loss
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import average_precision_score
import pickle

def load_data(database_filepath):
    """load data from database

    Args:
    database_filepath: str. Filepath of database file

    Returns:
    X:Pandas Dataframe. input message data
    Y:Pandas Dataframe. output labels
    category_names:Pandas Series. list of category names
    """
    # read in file
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table("cleaned_data",engine)
    
    # define features and label data
    X = df["message"]
    Y = df.iloc[:,4:]
    
    # get category names
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """ Normalize,tokenize and lemmantize the text
    
     Args:
    text: str. text message 

    Returns:
    df:List. list of tokenized words
    """
 
    text = text.lower() # lower the capital case
    text = re.sub(r"[^a-zA-Z0-9]"," ",text) #remove punctuation symbols
    words = word_tokenize(text) # tokenize into words
    
    # lemmantize the words and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    words_lem = [lemmatizer.lemmatize(word) 
                 for word in words if word not in stop_words]
    return words_lem


def build_model():
    """ define text processing & model pipeline and return 
    grid search model object.
    """
    
    # text processing and model pipeline
    pipeline = Pipeline([
    ("vect",CountVectorizer(tokenizer = tokenize)),
    ("TfidfVect",TfidfTransformer()),
    ("clf",MultiOutputClassifier(DecisionTreeClassifier
                                 (random_state=42))) ])
    
    # define parameters for for GridSearchCV
    parameters = {'clf__estimator__max_depth': [10,20]}
    
    # define scoring metrics
    scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average = 'macro'),
           'recall': make_scorer(recall_score, average = 'macro'),
           'f1_score': make_scorer(f1_score, average = 'macro')}
    
    # create grid search object and return as final model pipeline
    CV = GridSearchCV(estimator = pipeline,scoring = scoring ,
                  param_grid = parameters,refit ="f1_score"
                  ,cv=3,return_train_score=True)
    return CV
  
    
def evaluate_model(model, X_test, Y_test, category_names):
    """ provide the classification report"""
    
    Y_pred = model.predict(X_test)
    
    for i in range(0,category_names.size):
        target_names = [category_names[i]]
        print(classification_report(Y_test.iloc[:,i].values,
                                    Y_pred[:,i],target_names=target_names))

def save_model(model, model_filepath):
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print(model)
        
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