# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load two datasets and merge them.
    Args:
    messages_filepath: str. Filepath of messages data
    categories_filepath: str. Filepath of categories data
    
    Returns:
    df:Pandas Dataframe. Combined dataset of messages and categories
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on=("id"))
    return df
    

def clean_data(df):
    """Clean the dateset by clearly defining Category Names,
       convert category columns values to binary, and drops duplicates. 

    Args:
    df: Pandas Dataframe. Combined dataset of messages and categories

    Returns:
    df: Pandas Dataframe. Cleaned version of dataset
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[[0],:]
    
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda series : series[0][0:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype("str").apply(lambda str : str[-1])
    
        # convert column from string to binary values
        categories[column] = categories[column].astype("int64")
        categories[column] = categories[column].astype("bool")*1   
    
    #drop the original categories column from `df`
    df = df.drop("categories",axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    #Remove duplicates
    duplicate_rows = df.duplicated()
    if True in duplicate_rows:
        df = df[~duplicate_rows]
    
    return df
    
    
def save_data(df, database_filename):
    """Saves the data in SQL database.
    Args:
    df: Pandas Dataframe. Cleaned version of combined dataset of messages and categories.
    database_filename: str. Database filename to be used for storing data.
    """
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('cleaned_data', engine, index=False, if_exists="replace")


def main():
    """Load the datasets, clean data and save the processed data"""
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()