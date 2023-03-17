import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """Load messages & categories datasets & merge 
    
    inputs:
    messages_filepath: Filepath for file containing messages csv dataset
    categories_filepath: Filepath for file containing categories csv dataset
       
    outputs:
    df: dataframe. Dataframe containing merged data of messages & categories datasets
    """

    #Load Messages Dataset
    messages = pd.read_csv(messages_filepath)
    
    #Load Categories Dataset
    categories = pd.read_csv(categories_filepath)
    
    #Merge datasets
    df = messages.merge(categories, how = 'inner')
    
    return df

def clean_data(df):
    
    '''Cleans data
    inputs: df: created from merged messages & categories dataset
    
    outputs: cleaned dataframe of input df
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories by applying a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames
   
    # set each value to be the last character of the string & convert column from string to numeric
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column].str.replace('-', ''))
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, sort=False)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filepath):
    
    '''Saves clean dataframe to a SQLite database
    
    input: df: clean dataframe
           database_filename: file path for SQLite db
           
    output: SQLite database
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
     


def main():
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