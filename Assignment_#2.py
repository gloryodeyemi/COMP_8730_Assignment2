#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# 
# ### Glory Odeyemi
# 
# #### 6-Feb-2023

# ### Install libraries
# 
# You can skip this step if you already have these libraries installed.

# In[1]:


get_ipython().system('pip install pytrec-eval-terrier')
get_ipython().system('pip install nltk')


# ### Import libraries
# 
# This is an important step because some of the codes that depends on these libraries will give an error if the libraries are not imported.

# In[2]:


import nltk
import itertools
from utils.top_k_success import top_k_tokens, success_at_k, average_k
from utils.n_gram_model import tokenize_corpus, train_model, save_model, load_model


# ### Download Brown corpus
# 
# We use the news genre of the brown corpus to train our n-Gram language model in this project.

# In[3]:


nltk.download('brown')
nltk.download('punkt')


# In[4]:


from nltk.corpus import brown
# brown.categories()
brown_corpus_tokens = brown.words(categories='news')
print(brown_corpus_tokens[:10])


# In[5]:


print("Total number of tokens in the brown corpus news genre = ", len(brown_corpus_tokens))


# In[6]:


brown_corpus_sents = brown.sents(categories='news')
print(brown_corpus_sents[:2])


# In[7]:


print("Total number of sentences in the brown corpus news genre = ", len(brown_corpus_sents))


# ### Import Birkbeck corpus
# 
# Birkbeck spelling error corpus was used for this project. You can find it [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/0643).
# 
# The [APPLING1DAT.643](https://github.com/gloryodeyemi) file out of the Birkbeck spelling error corpus by Roger Mitton was used.

# In[8]:


birkbeck_data = []
with open('Data/APPLING1DAT.643', 'r') as file_data:
    for line in file_data:
        data = line.split()
        birkbeck_data.append(data)
birkbeck_data[:10]


# In[9]:


# clean corpus to remove line with $
for ind_list in birkbeck_data:
    for item in ind_list:
        if(item.startswith('$')):
            birkbeck_data.remove(ind_list)
        
birkbeck_data[:10]


# In[10]:


print("Total number of errored words in Birbeck corpus = ", len(birkbeck_data))


# ### Tokenizing the Brown corpus
# 
# The brown corpus has to be tokenized before we can use it to train our language models

# In[11]:


tokenized_corpus = tokenize_corpus(brown_corpus_sents)
print(tokenized_corpus[:2])


# ### Training the language model
# 
# We will train and save n-Gram language models using the tokenized brown corpus for n={1,2,3,5,10}

# In[12]:


n_list = [1, 2, 3, 5, 10]

for n in n_list:
    model = train_model(n, tokenized_corpus)
    save_model(n, model)


# ### Getting the top-k list of tokens and success at k
# 
# * Top-k list of tokens are the top most probable list of token that are retrieved by the language model.
# * For every incorrect word in the birkbeck_data corpus, top-k tokens are returned, where k={1,5,10}.
# * Success at k (s@k) measures whether the correct spelling of the word in the birkbeck_data corpus happens to be in the top-k (most probable) list of tokens that are retrieved by the language model.
# 
# **Sample test:** Two items in the birkbeck_data corpus will be used as test and sample result is shown.

# In[13]:


sample_test = birkbeck_data[50:52]
top_k_result = []

for n in n_list:
    model_loaded = load_model(n)
    print("--------------")
    print(f"{n}-gram model: \n--------------")
    for data_row in sample_test:
        res = top_k_tokens(data_row, model_loaded, tokenized_corpus)
        print(f"Top-k probability: {sample_test.index(data_row) + 1}", res)
        print("")
        top_k_result.append(res)
    
    success = success_at_k(top_k_result)
    print("Success at k: ", success)
    print("")    


# ### Evaluating all incorrect token in our birkbeck corpus and getting the average success at k for n={1,2,3,5,10}

# In[14]:


top_k_result = []

for n in n_list:
    model_loaded = load_model(n)
    print("--------------")
    print(f"{n}-gram model: \n--------------")
    for data_row in birkbeck_data:
        res = top_k_tokens(data_row, model_loaded, tokenized_corpus)
        top_k_result.append(res)
    
    success = success_at_k(top_k_result)
    avg = average_k(success)
    print("Average success at k: ", avg)

