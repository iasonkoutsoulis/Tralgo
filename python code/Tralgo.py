# This is the algorithm I'll use to do automated trading. 
from bs4 import BeautifulSoup as bs
from datetime import datetime
import json
import re
import requests

def link_collect(soop):
    all_linx = []
    for link in soop.find_all('a'):
        nlink = link.get('href')
        all_linx.append(nlink)
        all_linx = list(filter(lambda item: item is not None, all_linx))
    return all_linx

def year_collect(soop):
    years = []
    for timet in soop.find_all('time', {'class': 'fc-date-headline'}):
        years.append(re.findall(r'\d+', timet.string)[-1])
    years = list(dict.fromkeys(years))
    return years

def tl_collect(all_links, years):
    for yeart in years:
        expr = r'https:\/\/www\.theguardian\.com\/\S+\/' + yeart + r'\/\S+'
        text_links = []
        for link in all_links:
            if re.search(r'/all$', link):
                pass
            elif re.search(expr, link):
                text_links.append(link)
    return text_links

# main script 
bimon_arts = dict()
for page in range(287, 280, -1):
    print(str(page))
    # initialize using the front-page links
    url = 'https://www.theguardian.com/business/stock-markets?page=' + str(page) # total pages = 287
    html = requests.get(url).text
    soup = bs(html, 'lxml')

    # get all article links from the page we've opened (we use the year they include to identify them)
    all_links = link_collect(soup)
    years = year_collect(soup)
    text_links = tl_collect(all_links, years) 

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

        timet = subsoup.find('meta', {'property':'article:published_time'})
        try:
            fdate = timet['content']
        except Exception:
            pass
        dt_date = datetime.strptime(fdate, '%Y-%m-%dT%H:%M:%S.%fZ')
        art_date = str(dt_date.year) + '-' + str(dt_date.month)
        bimon = 'B2' if dt_date.day >= 15 else 'B1'
        
        if not (art_date + '-' + bimon) in bimon_arts:
            bimon_arts[art_date + '-' + bimon] = []
        else:
            pass

        article = [title]
        for textlink in subsoup.find_all('p'):
            article.append(textlink.string)
            article = list(filter(lambda item: item is not None, article))
            art_str = " ".join(article)
        if art_str in bimon_arts[art_date + '-' + bimon]:
            pass
        else:
            bimon_arts[art_date + '-' + bimon].append(art_str)

# instead of text files I'll try the JSON stuff now
with open('E:/Tralgo/articles/article_container.json', "w") as f:
    json.dump(bimon_arts, f)

# for a_bimon in bimon_arts:
#     with open('articles/articles_of_' + a_bimon + '.txt', 'a', encoding="utf-8") as f:
#         f.write(" ".join(bimon_arts[a_bimon]))



# to do:
# better and more content to scrape, less junk more substance?