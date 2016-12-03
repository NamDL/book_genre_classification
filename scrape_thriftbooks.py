import os
import sys
import re
import time
from bs4 import BeautifulSoup
import requests

# Scrape data from Thrift Books
# since book titles are specified individually in the link without specifying genre, download individual list pages to
# be scraped
# only scrape books for religion and humor
re_list = ['r_4.htm']
humor_list_2 = ["./h_1.htm", "./h_2.htm", "./h_3.htm", "./h_4.htm", "./h_5.htm", "./h_6.htm", "./h_7.htm", "./h_8.htm", './h_9.htm', './h_10.htm', './h_11.htm','./h_12.htm', './h_13.htm', './h_14.htm','./h_15.htm','./h_16.htm','./h_17.htm','./h_18.htm','./h_19.htm','./h_20.htm']


links = []
for f in humor_list_2:
    fh = open(f)
    soup = BeautifulSoup(fh, 'html.parser')
    bookblocks = soup.find_all('a', {'class':"BrowseEntry is-grid BookSlider-slide"})
    for book in bookblocks:
        links.append(book["href"])

counter = 1

root_dir = sys.argv[1]
for link in links:
    r = requests.get(link)
    content = r.text
    soup = BeautifulSoup(content, 'html.parser')
    title = soup.find('h1', {'class':'WorkMeta-title'}).get_text().strip()
    title = re.sub(r'\s+', ' ', title)
    description_div = soup.find('p', {'class' : 'WorkMeta-overview'})
    if description_div is not None:
        description = description_div.get_text()
        fh = open(os.path.join(root_dir, str(counter) + '.txt'), 'w', encoding='utf8')
        fh.write('%s\n' % title)
        fh.write('%s\n' % description)
        fh.close()
        counter += 1
        if counter % 50 == 0: # crawl delay to prevent overloading the server
            time.sleep(5)






