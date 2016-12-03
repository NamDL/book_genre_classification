#!/usr/bin/python

import os
import sys
import random
import requests
import re
from bs4 import BeautifulSoup
import string

count=0
urlSet=set()
reviewsUrl=set()

def processUrl(url):
    global reviewsUrl
    print(url)
    htmlText = requests.get(url)
    soups = BeautifulSoup(htmlText.text,"html.parser")
    for items in soups.find_all(attrs={'class':'product-image'}):
        for links in items.find_all('a'):
            reviewsUrl.add(links.get('href'))
    for url in reviewsUrl:
        printFile(url)

def printFile(url):
    global count    
    path="D:/544/Project/Data/Sci-fi"
    htmlText=requests.get("http://www.barnesandnoble.com"+url)
    soups=BeautifulSoup(htmlText.text,"html.parser")
    fileData=""
    title=""
    if(soups.title):
        title=soups.title.string
    for items in soups.find_all(attrs={'id':'Overview'}):
        for links in items.find_all('p'):            
            fileData+=re.sub('[^\x00-\x7F]','', links.getText())
    if fileData:
        count+=1
        title=(soups.find('h1',attrs={'itemprop':'name'})).get_text().strip()
        title=''.join(c for c in title if c not in '\/"?()!:{}<>')
        print(title)
        completePath = os.path.join(path, title[1:20]+".txt")
        open(completePath, "w").write(fileData)            


Urllist=[]
#51
for i in range(52,60):
    print(i)
    mainUrl="http://www.barnesandnoble.com/b/books/science-fiction-fantasy/_/N-1sZ29Z8q8Z180l?Nrpp=20&Ns=P_Sales_Rank&page="+str(i)
    mainHtmlText = requests.get(mainUrl)
    '''path="D:/544/Project/Data/Romance"
    completePath = os.path.join(path, str(count)+".txt")
    print(completePath)
    open(completePath, "w").write(mainHtmlText.text)'''
    #soup = BeautifulSoup(mainHtmlText.text,"html.parser")
    '''#processUrl(mainUrl)    
    urlSet.add(mainUrl)
    for item in soup.find_all(attrs={'class':'search-pagination'}):
        for link in item.find_all('a'):
            #Urllist.append(link.get('href'))
            urlSet.add(link.get('href'))

    for url in urlSet:
        print(url)'''
    processUrl(mainUrl.strip())
    #print(count)
