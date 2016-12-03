import sys
import re
import os
from bs4 import BeautifulSoup
import requests

# Scrape data from goodreads lists of books
# since structure is not easily discernible from the link, download individual pages to be scraped
humor_list = ["./h_l_6.htm", "./h_l_7.htm", "./h_l_8.htm", './h_l_9.htm', './h_l_10.htm']
romance_list = ["./r_l_1.htm", "./r_l_2.htm", "./r_l_3.htm", "./r_l_4.htm", "./r_l_5.htm", "./r_l_6.htm", "./r_l_7.htm", "./r_l_8.htm", './r_l_9.htm', './r_l_10.htm']
religion_list = ["./re_l_1.htm", "./re_l_2.htm", "./re_l_3.htm", "./re_l_4.htm", "./re_l_5.htm", "./re_l_6.htm", "./re_l_7.htm", "./re_l_8.htm", './re_l_9.htm', './re_l_10.htm', './re_l_11.htm']
sci_fi_list = ["./s_l_1.htm", "./s_l_2.htm", "./s_l_3.htm", "./s_l_4.htm", "./s_l_5.htm", "./s_l_6.htm", "./s_l_7.htm", "./s_l_8.htm", './s_l_9.htm', './s_l_10.htm', './s_l_11.htm','./s_l_12.htm', './s_l_13.htm']
mystery_list = ["./m_l_1.htm", "./m_l_2.htm", "./m_l_3.htm", "./m_l_4.htm", "./m_l_5.htm", "./m_l_6.htm", "./m_l_7.htm", "./m_l_8.htm", './m_l_9.htm', './m_l_10.htm', './m_l_11.htm','./m_l_12.htm', './m_l_13.htm', './m_l_14.htm']

# collect all links from the given set of pages
links = []
for f in mystery_list:
    fh = open(f)
    soup = BeautifulSoup(fh, 'html.parser')
    bookblocks = soup.find_all('a', {'class':"bookTitle"})
    for book in bookblocks:
        links.append(book["href"])

counter = 1

root_dir = sys.argv[1]
for link in links:
    r = requests.get(link)
    content = r.text
    soup = BeautifulSoup(content, 'html.parser')
    title = soup.find(id='bookTitle').text.strip()
    title = re.sub(r'\s+', ' ', title)
    description_div = soup.find(id='description')
    if description_div is not None:
        spans = description_div.find_all('span')
        if len(spans) > 1: # if the book has a description
            description = spans[1].get_text()
            fh = open(os.path.join(root_dir,str(counter) + '.txt'), 'w', encoding='utf8')
            fh.write('%s\n' % title)
            fh.write('%s\n' % description)
            fh.close()
            counter += 1





