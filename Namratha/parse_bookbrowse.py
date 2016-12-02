#!/usr/bin/python

import os
import sys
import random
import requests
import re
from bs4 import BeautifulSoup

count=0
Urllist=set()

def processUrl(url):
    reviewsUrl=set()
    print(url)
    htmlText = requests.get(url)
    soups = BeautifulSoup(htmlText.text,"html.parser")
    for items in soups.find_all(attrs={'class':'display_block'}):
        for links in items.find_all('a'):
            reviewsUrl.add(links.get('href'))
    for uniqueUrl in reviewsUrl:
        printFile(uniqueUrl)

def printFile(url):
    global count
    path="D:/544/Project/Data/Sci-fi"
    htmlText=requests.get(url)
    soups=BeautifulSoup(htmlText.text,"html.parser")
    fileData=""
    title=soups.title.string
    for items in soups.find_all(attrs={'id':'summary'}):
        for links in items.find_all('p'):            
            fileData+=re.sub('[^\x00-\x7F]','', links.getText())    
    if fileData:
        title=soups.find(attrs={'class':'left_column'})
        title=(title.find('i')).get_text()
        title=re.sub(r'[/\:*?"<>]', r'', title)
        count=count+1
        completePath = os.path.join(path, title+".txt")
        open(completePath, "w").write(fileData)            

'''def obtainPages(url):
    global Urllist
    pageList=set()
    htmlText = requests.get(url)
    soups = BeautifulSoup(htmlText.text,"html.parser")
    for items in soups.find_all(attrs={'class':'search-pagination'}):
        for links in items.find_all('a'):
            Urllist.add(link.get('href'))
            obtainPages(link.get('href'))'''

mainUrl="https://www.bookbrowse.com/browse/index.cfm/category_number/16/speculative-scifi-fantasy-alt-history"
mainHtmlText = requests.get(mainUrl)
'''path="D:/544/Project/Data/Humour"
completePath = os.path.join(path, str(count)+".txt")
print(completePath)
open(completePath, "w").write(mainHtmlText.text)'''
soup = BeautifulSoup(mainHtmlText.text,"html.parser")
processUrl(mainUrl)
for item in soup.find_all(attrs={'class':'pager'}):
    for link in item.find_all('a'):
        Urllist.add(link.get('href'))

for url in Urllist:
    processUrl(url.strip())

print(count)
