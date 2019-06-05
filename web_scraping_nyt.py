# update 06/05/2019

import requests
from bs4 import BeautifulSoup # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#css-selectors
import re # https://docs.python.org/3/library/re.html
import pandas as pd

def most_frequent(List):  # https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    return max(set(List), key = List.count) 

def data_mining_step1(start_year = 1851, end_year = 2018):
    urls = []
    for y in range(start_year, end_year + 1): # NEED TO MODIFY
        url = 'https://spiderbites.nytimes.com/%s/'%(y) # NEED TO MODIFY
        print(url)

        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
    #     print(soup.prettify())

        pages = soup.find_all(name = 'a', string = re.compile("Part")) # NEED TO MODIFY
        for p in pages:
            link = str(re.findall(r'[^/]+html', p['href'])[0]) # NEED TO MODIFY
            urls.append(url+link)
    #         print(url+link)
    
    return urls

def data_mining_step2(urls):
    yyyymm, links, titles = [], [], []
    for i, url in enumerate(urls):
        if i%100 == 0:
            print('%s / %s: '%(i,len(urls)), url)

        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
#         print(soup.prettify())
        (y,m) = re.findall(r'articles_(\d{4})_(\d{2})_\d+.html', url)[0] # NEED TO MODIFY, extract year and month

        pages = soup.select('ul#headlines li a') # NEED TO MODIFY
        for p in pages:
            # MEMORY ISSUE SOLVED: https://stackoverflow.com/questions/11284643/python-high-memory-usage-with-beautifulsoup
            link = str(p['href']) # NEED TO MODIFY
            title = str(p.string) # NEED TO MODIFY

            links.append(link)
            titles.append(title)
            yyyymm.append(int(y+m))
        #     print(link, title)
        
    return yyyymm, links, titles

def data_mining_step3(urls, url_links = [], yyyymmdd = [], authors = [], titles = [], bodies = [], START_INDEX = 0):
#     for i, url in enumerate(urls):
    for i in range(START_INDEX, len(urls)):
        url = urls[i]
        if (i%10 == 0) | (i == START_INDEX):
            print('%s / %s: '%(i,len(urls)), url)
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
#         print(soup.prettify())
        
        if (soup.find('html') is not None) and (len(soup.find_all(name='meta', attrs={'name':'errorpage'})) == 0) and ('http://theater.nytimes.com/mem/theater/treview.html' not in url) and ('http://movies.nytimes.com/movie/review' not in url): 
            #---- url_link -----------------------------------
            if len(re.findall(r'/(\d{4})/(\d{2})/(\d{2})/', url)) > 0:
                url_links_0 = url
            else:
                url_link = soup.find_all(name='meta', attrs={'property':'og:url'}) # ex) <meta property="og:url" content="https://www.nytimes.com/1852/01/31/archives/financial.html" />
                url_links_0 = str(url_link[0]['content'])
                
            #---- date -----------------------------------
            if len(re.findall(r'/(\d{4})/(\d{2})/(\d{2})/', url_links_0)) > 0:
                (y,m,d) = re.findall(r'/(\d{4})/(\d{2})/(\d{2})/', url_links_0)[0] 
            else:
                date = soup.find_all(name='meta', attrs={'itemprop':'datePublished'}) # ex) <meta data-rh="true" property="article:published" itemprop="datePublished" content="2018-06-19T20:25:49.000Z"/>
                (y,m,d) = re.findall(r'(\d{4})-(\d{2})-(\d{2})', date[0]['content'])[0] 
            yyyymmdd_0 = int(y+m+d)
            
            #---- author -----------------------------------
            author = soup.find_all(name='meta', attrs={'name':['byl', 'author']})
            authors_0 = re.findall('(?:^[Bb][Yy])?\s*(.*)', author[0]['content'])[0] # remove 'By ...', if any

            #---- title -----------------------------------
            title = soup.find_all(name='meta', attrs={'property':'og:title'})
            if len(title) == 0:
                title = soup.find_all(name='meta', attrs={'name':'hdl'}) # <meta name="hdl" content="No Rest for a Feminist Fighting Radical Islam">
            titles_0 = str(title[0]['content'])

            #---- story body -----------------------------------
            paras = soup.find_all(name = 'p')
            l_p_attrs_keys = [list(p.attrs.keys()) for p in paras]
            mf_p_attrs_key = most_frequent_list_of_list(l_p_attrs_keys) # most frequent <p> tag attribute key

            class_names = [p['class'][0]for p in paras if 'class' in p.attrs.keys()] # extract class names from the tags
            if (len(class_names) > 0) and ('story-body-text' in class_names):
                p_attrs_key = 'class'
                p_attrs_value = 'story-body-text'
            elif (len(class_names) > 0):
                p_attrs_key = 'class'
                p_attrs_value = most_frequent(class_names) # the most frequent class name
            else:
                p_attrs_key = 'itemprop'
                p_attrs_value = 'articleBody'

            bodies_0 = ''; bodies_1 = ''
            for p in paras:
                if (p_attrs_key in p.attrs.keys()) and (p_attrs_value in p[p_attrs_key]): # extract text only from the main story body
                    bodies_0 = bodies_0 + p.get_text() + '\n'
                if (mf_p_attrs_key == []) and (list(p.attrs.keys()) == []): 
                    bodies_1 = bodies_1 + p.get_text() + '\n'

            if (len(bodies_1) > len(bodies_0)) and (len(bodies_1) > 200): # replace the story body if the condition is met
                bodies_0 = bodies_1
                
            #-- add all ----------------------------------        
            url_links.append(url_links_0)
            yyyymmdd.append(yyyymmdd_0)
            authors.append(authors_0)
            titles.append(titles_0)
            bodies.append(bodies_0)
        else:
#             print('Nothing added because of one or more errors: ', url)
            url_links.append('')
            yyyymmdd.append('')
            authors.append('')
            titles.append('')
            bodies.append('')            
            
    return url_links, yyyymmdd, authors, titles, bodies

#---------------------------------------------------------------------------------
# start_year = 1851; end_year = 2018
start_year = 2018; end_year = start_year
grp_urls = data_mining_step1(start_year, end_year)
yyyymm, urls, titles = data_mining_step2(grp_urls) # it takes about 2 hours

url_links, yyyymmdd, authors, titles, bodies = [], [], [], [], [] # initialize

#---------------------------------------------------------------------------------
# Sometimes, this part fails to retrieve data from the website. In this case, just run this part again.
# This part is designed to resume the job from where it ended for any reason. 
url_links, yyyymmdd, authors, titles, bodies = data_mining_step3(urls, url_links, yyyymmdd, authors, titles, bodies, START_INDEX = len(url_links))

#---------------------------------------------------------------------------------
# The code below allows to skip an article by filling the lists with empty data
# url_links.append('')
# yyyymmdd.append('')
# authors.append('')
# titles.append('')
# bodies.append('')

#---------------------------------------------------------------------------------
df = pd.DataFrame({'url':url_links, 'date':yyyymmdd, 'author':authors, 'title':titles, 'body':bodies})
df.to_csv('NYT_news_articles_%s-%s.csv'%(start_year, end_year), index = False)
