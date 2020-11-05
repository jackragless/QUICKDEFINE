#!/usr/bin/env python
# coding: utf-8

# In[46]:


#required libraries
import pickle
import pandas as pd
from cleantext import clean
from nltk.stem import WordNetLemmatizer 
import re
import nltk
import numpy
import random
from gensim.summarization.summarizer import summarize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag, map_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
words = set(nltk.corpus.words.words())
new_stopwords = set(stopwords.words('english')) - {'not'}

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")


# In[62]:


#importing site objects, const_parse_data etc
with open("mined_sites_0_30.txt", "rb") as fp:   # Unpickling
    all_sites = pickle.load(fp)
with open("mined_sites_filtered_0_30.txt", "rb") as fp2:   # Unpickling
    filt_sites = pickle.load(fp2)
with open("syntactic_analysis/constituency_parse_dict_top.txt", "rb") as fp3:   # Unpickling
    const_parse_lvl1 = pickle.load(fp3)[0]
const_parse_lvl2 = list(pd.read_csv("2_lvl_const_parse.csv")['cons_parse'])
with open("kw_clean.txt", "rb") as fp4:   # Unpickling
    keyword = pickle.load(fp4)


# In[49]:


#returns true if string does not contain any foreign chars
def no_foreign(sent):
    condition = True
    for x in str(sent).split():    
        if not x.isalnum():
            condition = False
    return condition


# In[157]:


#returns true if string contains keyword towards start of sentence
def candidate(sent, kw):
    if clean(kw, no_punct=True, lang="en") in sent[0:30+len(kw)]:
#     if clean(kw, no_punct=True, lang="en") in sent:
        return True
    else:
        return False


# In[158]:


#removes all content within/including [] and () brackets
def regex(text):
    clean = re.sub(r'\[[^\]]*\]', '', text)
    clean = re.sub(r'\([^)]*\)', '', clean)
    return clean


# In[159]:


#lematises sentence / removes stopwords
def lematize(text,lemat_bool): #only returns lemmatised string if bool == True
    if lemat_bool == True:
        clean_sent = clean_sent.split()
        wl = WordNetLemmatizer()
        clean_sent = [wl.lemmatize(word) for word in clean_sent if not word in new_stopwords]
        clean_sent = ' '.join(clean_sent)
        return clean_sent
    
    else:
        return text


# In[160]:


#combine all of above functions to clean an array of site objects
def clean_driver(df, lemat_bool):
    corpus = []
    for site in df:
        temp_arr = []
        
        text = regex(site['text'])
        sent = nltk.sent_tokenize(text) 
        
        for j in range(len(sent)):
            
            clean_sent = clean(sent[j], no_punct=True, lang="en")
            clean_sent = lematize(clean_sent, lemat_bool)
              
            if no_foreign(clean_sent) and candidate(clean_sent, site['kw']) and len(clean_sent.split())<=60:
                temp_arr.append(clean_sent)
        
        corpus.append({'kwn':site['kwn'], 'kw':site['kw'], 'link':site['link'], 'sent':temp_arr} )
          
    return corpus


# In[161]:


all_sites_clean = clean_driver(all_sites, False)
filt_sites_clean = clean_driver(filt_sites, False)


# In[162]:


#random definiton model (used as benchmark to compare to other models)
def random_filter(master, kwn):
    final = []

    for i in master:
        if i['kwn'] == kwn:
            for j in i['sent']:
                final.append(j)

    if len(final)>=10:
        randarr = []
        for i in range(10):
            randarr.append(final[random.randint(0,len(final)-1)])
        return randarr
    else:
        return final


# In[173]:


#simplifies AllenNLP constituency parse object into 'lvl1 /// lvl2'
def tree_lvl1(main):
    main = main['hierplane_tree']['root']['children']
    final = ''
    for i in range(len(main)):
        final += main[i]['nodeType'] + ' '
    return final.strip()

def tree_lvl2(main):
    main = main['hierplane_tree']['root']['children']
    final = ''
    for i in range(len(main)):
        if 'children' in main[i]:
            for j in range(len(main[i]['children'])):
                final += main[i]['children'][j]['nodeType'] + ' '
        else:
            final += '?' + ' '
    return final.strip()


def const_parse_filter(string, lvl_num):
    temp_const = predictor.predict(sentence = string)
    if lvl_num == 1:
        if tree_lvl1(temp_const) in const_parse_lvl1:
            return True
        else:
            return False
    
    if lvl_num == 2:
        if tree_lvl1(temp_const) + ' /// ' + tree_lvl2(temp_const) in const_parse_lvl2:
            return True
        else:
            return False


# In[174]:


#drives const_parse functions above
def const_parse_driver(master, kwn, lvl_num):
    final = []
    count = 0
    for i in master:
        print(int(count/16)*100)
        if i['kwn'] == kwn:
            for j in i['sent']:
                if const_parse_filter(j, lvl_num):
                    final.append(j)
        count += 1
    return final


# In[176]:


#returns nost "important" string from an array (Gensim)
def textrank(para, ratio):
    if len(para) >=2:
        temp_para = ''
        for i in para:
            temp_para += i + '. '
        return summarize(temp_para, ratio=ratio, word_count=None, split=True)
    else:
        return para


# In[ ]:




