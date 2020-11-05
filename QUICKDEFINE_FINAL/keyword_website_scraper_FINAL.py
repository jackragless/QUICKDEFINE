#!/usr/bin/env python
# coding: utf-8

# In[1]:


#required packages
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import time
import urllib.parse 
import re

#reduces warnings output during requests without verification
requests.packages.urllib3.disable_warnings()
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += 'HIGH:!DH:!aNULL'
try:
    requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST += 'HIGH:!DH:!aNULL'
except AttributeError:
    pass


# In[2]:


#clean original concept keyword CSV file into array

df_orig = pd.read_csv("dataframe.csv")
df_clean = []

for i in df_orig["Link"]:
    clean = urllib.parse.unquote(str(i))
    clean = clean.replace('wiki/','').replace('_',' ').lower()
    df_clean.append(clean)
    
    
df_clean = df_clean[0:300]
    
df_clean


# In[3]:


# with open("kw_clean.txt", "wb") as fp:   #Pickling
#     pickle.dump(df_clean, fp)


# In[4]:


#request URL function; can handle various connection errors
connect_fail_cause = []
def extract_html(link):
    try:
        resp = requests.get(link, verify=False, timeout=15) #will fail after 15secs for efficiency
        soup = BeautifulSoup(resp.content, "html.parser")
        connect_fail_cause.append(0)
        return soup
#         return soup2
    except requests.ConnectionError:
        connect_fail_cause.append(1)
        return "CONNECT_FUBAR"
    except requests.Timeout:
        connect_fail_cause.append(2)
        return "CONNECT_FUBAR"
    except requests.TooManyRedirects:
        connect_fail_cause.append(3)
        return "CONNECT_FUBAR"


# In[5]:


#extract text from website html

def html_to_text(html):
    
    final = ''
    
    if isinstance(html,str)==False:
        text = html.find_all(['p','li']) #extracted tag types


        for p in text:
            final += p.text

        final = final.replace('\n',' ').replace('\r',' ').replace('\t','') #remove useless newline etc chars
        final = re.sub(' +', ' ', final) #remove space greater than one char length
    
    return final




# In[6]:


#exclude site objects empty / insufficient length

def not_empty(site_obj):
    if site_obj['soup'] == 'CONNECT_FUBAR':
        return False
    elif not site_obj['soup']:
        return False
    elif (len(site_obj['raw_text'])>=120) == False:
        return False
    else:
        return True


# In[7]:


single_page_time = []
google_page_time = []


# In[8]:


#drives previous functions to gather HTML/text of x pages for a given keyword into an array
def fetch_pages(query, pages_per_kw):
    print('\nnew_kw_query')
    start_time_google = time.time()
    keyword = query
    query = query.replace(' ', '+')
    URL = f"https://google.com/search?q={query}&num=24" #set number of results as num
    headers = {"user-agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"}
    resp = requests.get(URL, headers=headers) #headers ensure Google server consistently accepts requests
    google_page_time.append(time.time()-start_time_google)
    
    if resp.status_code == 200:
        final = []
        soup = BeautifulSoup(resp.content, "html.parser")
        print('new_kw_query_success')
        for g in soup.find_all('div', class_='rc'): #div.rc element contains search result URLs
            if not g == False and len(final)<pages_per_kw: #this condition ensures each keyword consistently contains x pages
                start_time_single = time.time()
                title = g.find('h3').text
                link = g.find('a')['href']
                print(link)
                soup2 = extract_html(link)
                print('finished_extracting')
                raw_text = html_to_text(soup2)
                item = { #site object
                    "kw": keyword,
                    "title": title,
                    "link": link,
                    "soup": soup2,
                    "raw_text": raw_text
                        }
                
                if not_empty(item):
                    final.append(item)
                single_page_time.append(time.time()-start_time_single)
                print('site_query_complete')
    
    return final


# In[9]:


#execute fetch_pages() here
master = []
start = 0
stop = 30
count = 0
x = df_clean[start:stop]
for kw in x: #NOTE: master array is nested in two layers: keyword---> site
    master.append(fetch_pages(kw, 16)) #set number of kw here
    count+=1
    print(int((count/(stop-start))*100),'% complete') #prints % progress of scraping


# In[10]:


result = []
result.append(connect_fail_cause)
result.append(single_page_time)
result.append(google_page_time)


# In[11]:


# with open("runtime_50_100.txt", "wb") as fp:
#     pickle.dump(result, fp)


# In[12]:


#Example
master[0][0]


# In[13]:


#flatten nested structure in order to pickle; hence kwn
FINAL = []
for i in range(len(master)):
    for index in (range(len(master[i]))):
        item = {
            "kwn": i,
            "kw": master[i][index]["kw"],
            "link": master[i][index]["link"],
            "soup": str(master[i][index]["soup"]), #pickle cannot accept bs4 object, convert to string
            "text": master[i][index]["raw_text"]
        }
        FINAL.append(item)


# In[15]:


#export site objects to file for heuristics selection phase
with open("mined_sites_0_30.txt", "wb") as fp:
    pickle.dump(FINAL, fp)


# In[ ]:




