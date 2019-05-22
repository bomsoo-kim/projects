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

        #---- url_link -----------------------------------
        if len(re.findall(r'nytimes.com/(\d{4})/(\d{2})/(\d{2})/', url)) > 0:
            url_links_0 = url
        else:
            url_links_0 = str(soup.html['itemid'])

        #---- date -----------------------------------
        (y,m,d) = re.findall(r'nytimes.com/(\d{4})/(\d{2})/(\d{2})/', url_links_0)[0]
        yyyymmdd_0 = int(y+m+d)

        #---- author -----------------------------------
        author = soup.find_all(name='meta', attrs={'name':['byl', 'author']})
        authors_0 = re.findall('(?:^[Bb][Yy])?\s*(.*)', author[0]['content'])[0] # remove 'By ...', if any

        #---- title -----------------------------------
        title = soup.find_all(name='meta', attrs={'property':'og:title'})
        titles_0 = str(title[0]['content'])
        
        #---- story body -----------------------------------
        paras = soup.find_all(name = 'p', class_ = re.compile('.+')) # find all tags with "p + class=..."

        class_names = [p['class'][0]for p in paras] # extract class names from the tags

        if 'story-body-text' in class_names: # determine the class name for the main story body
            story_body_class_name = 'story-body-text'
        else:
            story_body_class_name = most_frequent(class_names) # the most frequent class name

        bodies_0 = ''
        for p in paras:
            if story_body_class_name in p['class']: # extract text only from the main story body
                bodies_0 = bodies_0 + p.get_text() + '\n'
                
        #-- add all ----------------------------------        
        url_links.append(url_links_0)
        yyyymmdd.append(yyyymmdd_0)
        authors.append(authors_0)
        titles.append(titles_0)
        bodies.append(bodies_0)
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
df = pd.DataFrame({'url':url_links, 'date':yyyymmdd, 'author':authors, 'title':titles, 'body':bodies})
df.to_csv('NYT_news_articles_%s-%s.csv'%(start_year, end_year), index = False)
