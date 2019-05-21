import requests
from bs4 import BeautifulSoup # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#css-selectors
import re # https://docs.python.org/3/library/re.html
import pandas as pd

#------------------------------------------------------------------
def text_mining_step1():
    urls = []
    for y in range(1851, 2019): # NEED TO MODIFY
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
urls = text_mining_step1()

#------------------------------------------------------------------
def text_mining_step2(urls):
    links = []
    titles = []
    for i, url in enumerate(urls):
        if i%100 == 0:
            print('%s / %s: '%(i,len(urls)), url)

        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
        # print(soup.prettify())

        pages = soup.select('ul#headlines li a') # NEED TO MODIFY
        for p in pages:
            # MEMORY ISSUE SOLVED: https://stackoverflow.com/questions/11284643/python-high-memory-usage-with-beautifulsoup
            link = str(p['href']) # NEED TO MODIFY
            title = str(p.string) # NEED TO MODIFY

            links.append(link)
            titles.append(title)
        #     print(link, title)
        
    return links, titles
links, titles = text_mining_step2(urls) # it takes about 2 hours
