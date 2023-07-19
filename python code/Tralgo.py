# This is the algorithm I'll use to do automated trading. 
import requests
import re
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime

bimon_arts = dict()
for page in range(287, 286, -1):
    # initialize using the front-page links
    url = 'https://www.theguardian.com/business/stock-markets?page=' + str(page) # total pages = 287
    html = requests.get(url).text
    soup = bs(html, 'lxml')

    # get all article links from the page we've opened (we use the year they include to identify them)
    all_links = []
    for link in soup.find_all('a'):
        nlink = link.get('href')
        all_links.append(nlink)
        all_links = list(filter(lambda item: item is not None, all_links))

    years = []
    for timet in soup.find_all('time', {'class': 'fc-date-headline'}):
        years.append(re.findall(r'\d+', timet.string)[-1])
    years = list(dict.fromkeys(years))    

    for yeart in years:
        expr = r'https:\/\/www\.theguardian\.com\/\S+\/' + yeart + r'\/\S+'
        text_links = []
        for link in all_links:
            if re.search(expr, link):
                text_links.append(link)


    # now we open all of the articles on the page and collect them into our bimonthly datasets
    # we create a dictionary/log entry which holds all text for a span of 15 days.
    for tlink in text_links:
        subhtml = requests.get(tlink).text
        subsoup = bs(subhtml, 'lxml')
        texpr = r'^.*?(?= \||$)'
        try:
          title = re.search(texpr, subsoup.title.string).group(0)
        except Exception:
          pass

        timopub = []
        timet = subsoup.find('meta', {'property':'article:published_time'})
        fdate = timet['content']
        dt_date = datetime.strptime(fdate, '%Y-%m-%dT%H:%M:%S.%fZ')
        art_date = str(dt_date.year) + '-' + str(dt_date.month)
        bimon = 'B2' if dt_date.day >= 15 else 'B1'
        
        if not (bimon + '-' + art_date) in bimon_arts:
            bimon_arts[bimon + '-' + art_date] = [] 
        else:
            pass

        article = [title]
        for textlink in subsoup.find_all('p'):
            article.append(textlink.string)
            article = list(filter(lambda item: item is not None, article))
            art_str = " ".join(article)
        if art_str in bimon_arts[bimon + '-' + art_date]:
            pass
        else:
            bimon_arts[bimon + '-' + art_date].append(art_str)

for a_bimon in bimon_arts:
    with open('articles/articles_of_' + a_bimon + '.txt', 'a', encoding="utf-8") as f:
        f.write(" ".join(bimon_arts[a_bimon]))

# to do:
# better scraping for content, less junk more substance?