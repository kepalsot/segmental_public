#!/usr/bin/env python
# coding: utf-8

# ## modelling for segmental study 
# ## kepler ps 
# #### contact: kpalacio@princeton.edu 
# 
# ### table of contents 
# #### 1. 
# a. load packages
# 
# b. read in data 
# 
# c. format for modelling 
# 
# 
# #### 2
# a. output display 
# 
# b. modeling, output storage
# 
# 
# #### 3
# a. view topics and topic proportions matrices
# 
# b. test, label and export matrices 
# 

# # 1a. 

# In[81]:


# load packages 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# # 1b. 

# In[ ]:


# read in data 
corpus_allclips = pd.read_csv('/path-to-timestamps')
timestamps = pd.read_csv('/path-to-timestamps')


# # 1c. 

# In[ ]:


# model requires a list of strings as input
model_input = [str(sec) for sec in corpus_allclips['text']]
model_input


# # 2a. 
# 

# In[173]:


# output display
# takes in a fitted model instance, topic feature names, and the number of feature items to display
def display_topics(model, feature_names, num_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-num_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-num_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# # 2b. 

# In[1]:


# vectorize and store the corpus
vectorizer = CountVectorizer() # optional params: max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
vects_allclips = vectorizer.fit_transform(model_input)
feature_names_allclips = vectorizer.get_feature_names_out()

# set number of top words
n_top_words = 10
# fit lda on the corpus - 100 topics 
lda = LatentDirichletAllocation(n_components=100, random_state=0).fit(vects_allclips) # max_iter=5, learning_method='online', learning_offset=50.,

# store model output
topic_matrix_allclips = display_topics(lda, feature_names_allclips, n_top_words) # matrix storing topics (features), and each topic item's probability
tpm_allclips_array = lda.transform(vects_allclips) # topic distributions over each timestamp, stored as an array



# # 3a. 

# In[175]:


# view topic matrix 
topic_matrix_allclips


# # 3b.

# In[179]:


# test, label and export topic proportions matrix 
# turn tpm into a dataframe for readability
tpm_allclips = pd.DataFrame(tpm_allclips_array)


# In[182]:


tpm_allclips


# In[ ]:


# label tpm with the timestamps df 
tpm_labelled = pd.concat([timestamps_df, tpm_allclips], ignore_index=False, axis=1)


# In[ ]:


# check proportion sums by row 
# should be = or close to 1
for index, row in tpm_allclips.iterrows():
    print(np.sum(row))


# In[190]:


# check dimensions
tpm_allclips.shape
timestamps_df.shape


# In[133]:


# export the topic-proportions matrix!
tpm_allclips.to_csv(r'path')
# export tpm with timestamps & clip labels 
tpm_labelled.to_csv(r'path')

