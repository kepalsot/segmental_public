#!/usr/bin/env python
# coding: utf-8

# ## preprocessing for segmental study 
# ## kepler ps 
# #### contact: kpalacio@princeton.edu 
# 

# ### table of contents 
# #### 1. raw data preprocessing
# a. load packages, data, parse sheets
# 
# b. sort annotations by clip
# 
# c. check annotation lengths
# 
# d. combine each participant's annotations by clip
# 
# 
# #### 2. text preprocessing
# a. remove stop words 
# 
# b. rename columns
# 
# c. format for vectorization and modelling 
# 
# d. combine all clips
# 
# e. export timestamps, corpus
# 
# 

# # 1a. load packages, data, parse sheets

# In[11]:


# load packages 
import os
import pandas as pd
import datetime
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[12]:


# create dfs for each sheet (clip annotation) for each participant
# assumes data is in a .xlsx filetype 
def create_dataframes(folder_path, target_string):
    files = os.listdir(folder_path)
    dfs = {}
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file)
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            for sheet_name in sheet_names:
                if target_string not in sheet_name:
                    dfs[file[18:20] + sheet_name] = pd.read_excel(file_path, sheet_name)
    return dfs


# In[13]:


# read in raw data
folder_path = r'C:\Users\kp3434\OneDrive - Princeton University\Desktop\segmental\segmental\data\annotations'  
target_character_string = '-'  
raw_annotations = create_dataframes(folder_path, target_character_string)


# ## 1b. Sort annotations by clip 

# In[14]:


# dict to hold dfs 
all_clips = {f"clip{i}":{} for i in range(1, 10)}


# In[15]:


# sort annotations by clip
for key, item in raw_annotations.items():
    n = key[-1]
    if f"clip{n}" in key:
        all_clips[f"clip{n}"][key] = item


# ## 1c. Check annotation lengths for each clip 
# ### make sure annotation lengths are consistent within clips

# In[16]:


# get axis 0-lengths for each participant in each clip
# within-clip dfs should be the same length but between-clip lengths can vary 
# this approach is fine forthe number of clips & participants we have, but a systematic 
for clip, annotations in all_clips.items(): 
    for participant, seconds in annotations.items():
        print(participant + " length:", len(seconds))
    


# In[54]:


# isolate timestamps 
# doesn't include countdown timestamps 
timestamps = {}

# for each clip, collect timestamps
# only need timestamps for one participant's annotation
for clip, annotations in all_clips.items():
    for participant, seconds in annotations.items():
        timestamps[clip] = seconds.iloc[4:,0].reset_index(drop=True)
        break # only need one timestamp column per clip


# In[55]:


timestamps['clip3']


# ## 1d. Combine each participant's annotations by clip

# In[56]:


clips_sorted = {}
# combines all clips horizontally 
for clip, annotations in all_clips.items():    
    # concatenate for a single clips
    clips_sorted[clip] = pd.concat(all_clips[clip].values(), axis=1)
    clips_sorted[clip] = clips_sorted[clip].iloc[:,1::2] # remove timestamp repeats


# In[57]:


all_clips['clip1']


# # 2. Text preprocessing

# ## 2a. remove stop words 

# In[58]:


print(stopwords.words('english'))


# In[106]:


# get stop words 

# function for removing stop words and non-alphanumeric characters 
def text_preprocess(df, count):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_text = pd.DataFrame()

    for column in df.columns:
        preprocessed_text[column] = None

        for index, row in enumerate(df[column]):
#             # add datetime data to new df as-is
#             print("NEW CELL")
#             if type(row) == datetime.time:
#                 preprocessed_text.at[index, column] = row
#                 continue
                
            # tokenize single-row string
            word_tokens = word_tokenize(row)
            print("Raw tokens:",word_tokens)
            # lemmatize (break words down to lemma-level)
            lemmatized_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
            print("Lemmatized:", lemmatized_tokens)
            
            # filter out stopwords
            filter_stopwords = [w.lower() for w in lemmatized_tokens if not w.lower() in stop_words] # lower() input to compare to stop_words list
            filtered_list_strict = [s for s in filter_stopwords if s.isalnum()] # strict - only tokens with alphanumerics
            filtered_list_lax = [s for s in filter_stopwords if any(c.isalnum() for c in s)] # lax - any with at least one alphanumeric
            print("Stop words rm-STRICT + stemmed:",filtered_list_strict)
            print("Stop words rm-LAX + stemmed:",filtered_list_lax)
            # recombine tokens - will be tokenized again later in fit_transform()
            as_string_lax = ' '.join(filtered_list_lax)
            as_string_strict = ' '.join(filtered_list_strict) 

            preprocessed_text.at[index, column] = as_string_strict # chose to use strict filtering 
            
            preprocessed_texts[str(count)] = preprocessed_text
    return preprocessed_texts
        


# In[107]:


corpus_allclips


# ## 2b. rename columns

# In[108]:


# dictionary for all preprocessed clips 
preprocessed_texts = {}
count = 1

# text-preprocess all clips
for clip, df in clips_sorted.items():
    # rename columns
    colnames = [f'annotator{i}' for i in range(1, len(df.columns)+1)]
    df.columns = colnames
    preprocessed_texts = text_preprocess(df, count)
    count += 1


# In[109]:


# remove 3rd clip (practice clip)
# both for annotations and timestamps
preprocessed_texts.pop('3')
timestamps.pop('clip3')
preprocessed_texts.keys()


# In[110]:


preprocessed_texts
    


# ## 2c. format for vectorization and modelling 

# In[111]:


corpora = {} # use a dictionary to store the corpus for each clip 

# loop through all clips
for clip_key, clip_df in preprocessed_texts.items(): 
    # create corpus (dataframe) for all rows of a single clip
    corpus = pd.DataFrame() 

    # loop through each clip's rows; rows are individual documents
    for row in clip_df.values:
        # join all annotations across the row (one row : one timestamp) to create the 'document'
        row_text = pd.Series(' '.join(row)) # row[2:] to discard timestamp column; will reattach to model output
        # add the document to the corpus
        corpus = pd.concat([corpus, row_text], ignore_index=True) 
    
    corpus = corpus.iloc[4:,:].reset_index(drop=True) # remove countdown text 
    corpora[clip_key] = corpus # add clip-specific corpus to the corpora

    # create document (row-wise appending of text )
# concatenate entire df EXCEPT for timestamp column & similar stuff 
# specify clip-specific number of topics (directly depends on how many unique timestamps exist for each clip)
# fit lda on entire clip document 
# generte BOTH a topics matrix and a topic-proportions matrix


# In[51]:


pd.set_option('display.max_colwidth', None)


# In[70]:


# combine timestamps
timestamps_df = pd.concat(timestamps, axis = 0) # pd.concat(timestamps, axis=0, keys=range(len(timestamps)), names=['clip']) for new index levels
timestamps_df = timestamps_df.reset_index(level = 0)
timestamps_df = timestamps_df.rename(columns = {'level_0':'clip'})


# In[71]:


timestamps_df


# In[70]:





# ##  2d. combine all clips
# 

# In[158]:


# combine each corpus in the corpora into a single corpus 
corpus_allclips = pd.concat(corpora.values(), ignore_index=True)
corpus_allclips.columns = ['text']


# In[78]:


# Ensure combined len == to sum of len of each clip 
if len(corpus_allclips) == sum([len(data) for key, data in corpora.items()]): # use corpora, not preprocessed_text()
    print(len(corpus_allclips))

# check difference in clip lengths after dropping 
for name, df in preprocessed_texts.items():
    print(len(df), len(corpora[name]))


# # 2e. 

# In[ ]:


# save timestamps df
timestamps_df.to_csv('C:\\Users\\kp3434\\OneDrive - Princeton University\\Documents\\segmental\\data\\model output\\timestamps.csv')


# In[ ]:


# save corpus df 
corpus_allclips.to_csv('path')

