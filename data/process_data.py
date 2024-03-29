# import statements

import nltk
import sys
import pandas as pd
import re
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sqlalchemy import create_engine

# download necessary NLTK data
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

def load_data(messages_filepath, categories_filepath):
  """
  Load the data containing the messages and the
  categories, and merge on the common column 'id'
  Returns a single pandas Dataframe.

  Keyword arguments:
  messages_filepath -- filepath (including file name) 
                       of the file containing the messages
  categories_filepath -- filepath (including file name) 
                         of the file containing the message 
                         categories
  """
  messages = pd.read_csv(messages_filepath, encoding='UTF-8')
  categories = pd.read_csv(categories_filepath, encoding='UTF-8')
  return messages.merge(categories, on=['id'])


def clean_data(df):
  """
  Parse the single 'categories' column into the 36 distinct
  message category columns, name the resulting columns, and 
  clean the values, removing the category name from the cells
  and leaving only the numeric categorical value.

  Remove true duplicate rows.

  Returns a cleaned Dataframe.

  Keyword argument:
  df -- Dataframe requiring cleaning.
  """
  categories_new = df['categories'].str.split(pat=';', expand=True)
  row = categories_new.iloc[0,:]
  category_colnames = list(row.apply(lambda x: x[:-2]))
  categories_new.columns = category_colnames
  for column in categories_new:
    # set each value to be the last character of the string
    categories_new[column] = categories_new[column].str.slice(-1)
    
    # convert column from string to numeric
    categories_new[column] = categories_new[column].astype(int)

  df.drop(columns=['categories'], inplace=True)

  df = pd.concat([df, categories_new], axis=1)
    
  # drop duplicates
  df.drop_duplicates(inplace=True)
  return df

def save_data(df, database_filename):
  """
  Save cleaned Dataframe to a SQL Database table.

  Keyword arguments:
  df -- Cleaned Dataframe for export
  database_filename -- name of the database in which
                       table will be saved
  """
  engine = create_engine('sqlite:///' + database_filename)
  df.to_sql('messages_and_categories', engine, index=False)  


def main():
  """
  Executes following functions:

  1) load_data(messages_filepath, categories_filepath)
  2) clean_data(df)
  3) save_data(df, database_filename) 
  """
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