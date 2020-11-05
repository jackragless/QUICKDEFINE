#!/usr/bin/env python
# coding: utf-8

# In[28]:


#required libraries
import pickle
import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
import textstat
import readability
from textblob import TextBlob


# In[29]:


#import pickled files from previous stages
with open("mined_sites_0_30.txt", "rb") as fp1:
    import_data = pickle.load(fp1)
    
with open("kw_clean.txt", "rb") as fp2:
    import_kw = pickle.load(fp2)


# In[30]:


#convert flat array into a nested one
def build_master(master):
    master = pd.DataFrame(master)
    final = []
    count = 1
    kwn = 0
    temp = []
    while count<=master.shape[0]: #change
        item = {
            'kwn' : master['kwn'][count-1],
            'kw' : master['kw'][count-1].replace('+',' '),
            'link' : master['link'][count-1],
            'soup' : BeautifulSoup(master['soup'][count-1]),
            'text' : master['text'][count-1],
            'score' : 1
        }
        temp.append(item)
        
        if count % 16 == 0:
            final.append(temp)
            temp = []
            kwn+=1
            
        count+=1
        
    return final


# In[31]:


master = build_master(import_data)
del import_data


# In[32]:


#remove academic websites altogether; this function includes other unwanted sites
def academic(site):
    
    remove_list = ['.edu','.pdf','university','stackoverflow.com/questions','stackexchange.com/questions']
    
    if any(substring in site["link"] for substring in remove_list):
        return 0
    else:
        return site['score']


# In[33]:


#determine if site commercial if it contains 'Solutions' or 'Products' strings in its header HTML
def commercial(site):
    temp = str(site["soup"].select("ul")[0:2])
    temp = BeautifulSoup(temp, "html.parser")
    temp = temp.select('a')
    for k in temp:
        if 'Solutions' in k.text or 'Products' in k.text:
            return 1
            break
    return 0


# In[34]:


#determines esotericism of text; calculates probability of a word being another CS concept from the SO dataset
def esoteric(site):
    temp_score = 0
    
    for i in range(len(import_kw)):
        excl_list = import_kw[:i] + import_kw[i:]
        for k in excl_list:
            if k in site["text"].lower():               #NEW
                temp_score += 1
    
    temp_score /= (len(site["text"].split())+1)
                    
    return temp_score 


# In[35]:


def subjective(site):
    return TextBlob(site['text']).subjectivity


# In[36]:


def readable(site):
    return textstat.flesch_reading_ease(site["text"])


# In[37]:


#weights heuristics based on the number of stdev from mean of given keyword
def distribution(array, heuristic):
    temparr = []
    if heuristic == 'subjective' or heuristic == 'esoteric':
        stdr = np.mean(array, axis = None) + 2*np.std(array, axis = None)
        stdl = np.mean(array, axis = None) + np.std(array, axis = None) 
        for i in array:
            if i >= stdl and i <= stdr:
                temparr.append(1)
            elif i > stdr:
                temparr.append(2)
            else:
                temparr.append(0)
        
        
    elif heuristic == 'readable':
        stdr = np.mean(array, axis = None) - np.std(array, axis = None) 
        stdl = np.mean(array, axis = None) - 2*np.std(array, axis = None)
        for i in array:
            if i <= stdr and i >= stdl:
                temparr.append(1)
            elif i < stdl:
                temparr.append(2)
            else:
                temparr.append(0)
                
    return temparr


# In[38]:


#driver for all the above functions; directly subtracts from the scores in master array
def scoregen(heuristic, weight):
    for i in range(len(master)):
        temp_arr = []
        for j in range(len(master[i])):
            print(master[i][j]['link'])
            site = master[i][j]
            if heuristic == 'esoteric':
                temp_arr.append(esoteric(site))
            elif heuristic == 'subjective':
                temp_arr.append(subjective(site))
            elif heuristic == 'readable':
                temp_arr.append(readable(site))
        
        dist_arr = distribution(temp_arr,heuristic)
        dist_arr = [element * weight for element in dist_arr]
        
        print(dist_arr)
        
        for k in range(len(master[i])):
            master[i][k]['score'] -= dist_arr[k]
    return


# In[39]:


scoregen('esoteric',0.1)
scoregen('subjective',0.1)
scoregen('readable',0.1)


# In[40]:


#these factors are done separately as they do not use distribution; 
#if commercial subtract 0.1 
#if academic set score = 0; eliminating with certainty 
for i in range(len(master)):
    for j in range(len(master[i])):
        master[i][j]["score"] -= commercial(master[i][j])*0.1
        master[i][j]["score"] = academic(master[i][j])


# In[41]:


#sorts website objects by score and selects top x num; before creating a flat array to pickle
FINAL = []
for i in range(len(master)):
    temp_score = {}
    for j in range(len(master[i])):
        temp_score.update({j:master[i][j]["score"]})
    temp_score = dict(sorted(temp_score.items(), key=lambda x: x[1], reverse=True))
    temp_score2 = list(temp_score.keys())[0:8] 
    for index in temp_score2:
        item = {
            "kwn": i,
            "kw": master[i][index]["kw"],
            "link": master[i][index]["link"],
            "text": master[i][index]["text"]
        }
        FINAL.append(item)


# In[42]:


#exporting remaining website objects to local file
with open("mined_sites_filtered_0_30.txt", "wb") as fp:   #Pickling
    pickle.dump(FINAL, fp)


# In[ ]:




