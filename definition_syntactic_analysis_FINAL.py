#!/usr/bin/env python
# coding: utf-8

# In[79]:


#required libraries
import nltk
import pandas as pd
from pdfminer.high_level import extract_text
import pickle
import re
import os
from cleantext import clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag, map_tag
from nltk import RegexpParser
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")


# In[80]:


#uncomment to extract keywords from dictionary glossary pdf

# keyword = extract_text('kw_only.pdf')
# keyword = keyword.replace('\t',' ').replace('\x0c','')
# keyword = keyword[keyword.index('10 Base 2'):].lower().split('\n\n')
# pd.DataFrame(data = keyword).to_csv('kw_array.csv')


# In[81]:


#read in definition keywords previously extracted from dictionary glossary
keyword = pd.read_csv('kw_array.csv')
keyword = keyword.dropna(how='any')
keyword = list(keyword['0'])


# In[82]:


#read in definition sentences previously extracted from dictionary body
definition = open("raw_defs.txt", "r").read()
#clean various formatting characters
definition = definition.replace('\t',' ').replace('\x0c','').lower()
def_array = definition.split('\n\n')
for i in range(len(def_array)):
    def_array[i] = def_array[i].replace('\n', ' ')


# In[83]:


#remove definitions referring to acronyms of other definitions
z = 0
while z < len(def_array):
    if 'abbrev.' in def_array[z]:
        del def_array[z]
    z+=1


# In[84]:


#create array of definition objects by combining keywords and definition lists
master = []
for h in range(len(keyword)):
    temp = []
    for i in range(len(def_array)):
        if keyword[h] in def_array[i][0:len(keyword[h])]: #ensure def matches the kw
            excl_kw = def_array[i][def_array[i].index(keyword[h])+len(keyword[h])+1:] #cut off kw from def
            if excl_kw.find('.') != -1:
                excl_kw = excl_kw[0:excl_kw.index('.')]
            if len(excl_kw.split(" ")) < 5 and "see" in excl_kw: #exclude def if too short or refers to another definition
                break
            else:
                temp.append(excl_kw)
    if len(temp)==1:
        master.append( { 'kw':keyword[h], 'def':temp[0], 'def_clean':''} )


# In[85]:


#corpus-specific cleaning conditions
for i in range(len(master)):
    #for keywords with multiple definitions; extract the first only
    one_find = master[i]['def'].find('1.')
    two_find = master[i]['def'].find('2.')
    if one_find != -1 and two_find != -1:
        master[i]['def'] = master[i]['def'][one_find+2:two_find]
        #clean grammar
    master[i]['def_clean'] = re.sub(r'\([^)]*\)', '', master[i]['def'])
    master[i]['def_clean'] = clean(master[i]['def_clean'],no_punct=True, lang="en")


# In[86]:


#lematises / removes stopwords from given string
def lemat_sent(sent):
    corpus = []
    for i in sent:
        text_data = nltk.word_tokenize(i['def_clean'])
        wl = WordNetLemmatizer()
        text_data = [wl.lemmatize(word) for word in text_data if not word in stopwords.words('english')]
        text_data = ' '.join(text_data)
        corpus.append(text_data)
    return corpus


# In[87]:


lemat_sent = lemat_sent(master)


# In[88]:


#put all definition sentences in one string array
orig_sent = []
for i in range(len(master)):
    orig_sent.append(master[i]['def_clean'])


# In[89]:


#Part-Of-Speech tagger
def pos_tag(sent):
    final = []
    for i in range(len(sent)):
        split = nltk.word_tokenize(sent[i])
        pos = nltk.pos_tag(split)
        sim_pos = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos] #uses simpler universal tags
        pos_arr = [i[1] for i in sim_pos]
        final.append(pos_arr) 
    return final


# In[90]:


orig_sent_pos = pos_tag(orig_sent)
lemat_sent_pos = pos_tag(lemat_sent)


# In[93]:


#finds highest frequency POS sequences of x length
def find_pattern_adv(pos, num):
    final = []
    for i in range(len(pos)):
        
        cat_str = ''
        
        if len(pos[i]) > num:
            for j in range(num):
                cat_str += pos[i][j] + " "
                
            final.append(cat_str)

    return pd.Series(final).value_counts()   


# In[98]:


find_pattern_adv(orig_sent_pos, 8)


# In[19]:


#these functions are desgined to parse the heavily nested AllenNLP constituency parse array
#reduces to a single string per level

def tree_lvl1(main):
    main = main['hierplane_tree']['root']['children']
    final = []
    for i in range(len(main)):
        final.append(main[i]['nodeType'])
    return final

def tree_lvl2(main):
    main = main['hierplane_tree']['root']['children']
    final = []
    for i in range(len(main)):
        if 'children' in main[i]:
            for j in range(len(main[i]['children'])):
                final.append(main[i]['children'][j]['nodeType'])
        else:
            final.append('?')
    return final

def tree_lvl3(main):
    main = main['hierplane_tree']['root']['children']
    final = []
    for i in range(len(main)):
        if 'children' in main[i]:
            for j in range(len(main[i]['children'])):
                if 'children' in main[i]['children'][j]:
                    for k in range(len(main[i]['children'][j]['children'])):
                        final.append(main[i]['children'][j]['children'][k]['nodeType'])
                else:
                    final.append('?')
        else:
             final.append('?')
    return final

def tree_lvl4(main):
    main = main['hierplane_tree']['root']['children']
    final = []
    for i in range(len(main)):
        if 'children' in main[i]:
            for j in range(len(main[i]['children'])):
                if 'children' in main[i]['children'][j]:
                    for k in range(len(main[i]['children'][j]['children'])):
                        if 'children' in main[i]['children'][j]['children'][k]:
                            for m in range(len(main[i]['children'][j]['children'][k]['children'])):
                                final.append(main[i]['children'][j]['children'][k]['children'][m]['nodeType'])
                        else:
                            final.append('?')
                else:
                    final.append('?')
        else:
            final.append('?')
    return final

def build_levels(main):
    return {
        'l1':tree_lvl1(main),
        'l2':tree_lvl2(main),
        'l3':tree_lvl3(main),
        'l4':tree_lvl4(main)
    }


# In[24]:


#prepends keyword + "is" to definitions starting with DET
FINAL = []
for i in range(len(master)):
    FINAL.append( {'kw':master[i]['kw'],'sent':master[i]['def_clean'], 'pos':orig_sent_pos[i]} )
COMBINE = []
for i in range(len(FINAL)):
    if len(FINAL[i]['pos'])>=1 and FINAL[i]['pos'][0] == 'DET':
        COMBINE.append( {'sent':FINAL[i]['kw'] + ' is ' + FINAL[i]['sent'], 'pos':['NOUN','VERB']+FINAL[i]['pos']} )


# In[99]:


len(COMBINE)


# In[28]:


#builds constituency parse trees master array
levels_final = []
for i in range(len(COMBINE)):
    temp_const = predictor.predict(
    sentence= COMBINE[i]['sent']
    )
    levels_final.append(build_levels(temp_const))
    if i%30 == 0:
        print(int(i/len(COMBINE)*100))


# In[29]:


with open('constituency_parse_dict.txt', 'wb') as f:
    pickle.dump(levels_final, f)


# In[30]:


levels_final


# In[31]:


def pos_to_str(arr):
    final = ''
    for i in range(len(arr)):
        final += arr[i] + ' '
    return final.strip()


# In[32]:


levels_final_sorted = [[],[],[],[]]
for obj in levels_final:
    levels_final_sorted[0].append(pos_to_str(obj['l1']))
    levels_final_sorted[1].append(pos_to_str(obj['l2']))
    levels_final_sorted[2].append(pos_to_str(obj['l3']))
    levels_final_sorted[3].append(pos_to_str(obj['l4']))


# In[33]:


levels_final_sorted


# In[34]:


FINAL_LEVELS = []
for i in range(4):
    temp_levels = pd.DataFrame(levels_final_sorted[i])
    FINAL_LEVELS.append(temp_levels.value_counts()[temp_levels.value_counts()>=5].index.tolist())
    for j in range(len(FINAL_LEVELS[i])):
        FINAL_LEVELS[i][j] = FINAL_LEVELS[i][j][0]


# In[35]:


FINAL_LEVELS


# In[36]:


with open('constituency_parse_dict_top.txt', 'wb') as f:
    pickle.dump(FINAL_LEVELS, f)


# In[37]:


# final4 = []
# for h in range(4,10):
#     final3 = []
#     # num=5
#     for i in range(len(test)):

#         cat_str = ''

#         if len(test[i]['pos']) > h:
#             for j in range(h):
#                 cat_str += test[i]['pos'][j] + " "

#             final3.append(cat_str.strip())
            
#     final3 = pd.Series(final3)
#     final3 = final3.value_counts()[final3.value_counts()>2].index.tolist()
#     final4.append(final3)


# In[ ]:


# final4[2]


# In[ ]:


# with open('pos_dict.txt', 'wb') as f:
#     pickle.dump(final4, f)


# In[ ]:




