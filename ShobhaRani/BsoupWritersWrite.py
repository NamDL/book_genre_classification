from bs4 import BeautifulSoup
import urllib.request
import sys
import re

#Code to download the HTML of the given URL
#url = "http://www.writerswrite.com/books/excerpts/forgedinblood.htm"

#page = urllib.request.urlopen(url)

#soup = BeautifulSoup(page)
#print(soup.prettify())


#Finding all the links in the current genre
genreURL = "http://www.writerswrite.com/books/excerpts/romance/"
page = urllib.request.urlopen(genreURL)
genreSoup = BeautifulSoup(page)
count = 0
for link in genreSoup.find_all('a'):
	excerptURL = link.get('href')
	if(re.search('http://www.writerswrite.com/books/excerpts/',excerptURL)!=None):
		book = urllib.request.urlopen(excerptURL)
		bookSoup = BeautifulSoup(book)
		fName = "srd_romance_" + str(count) + ".txt"
		f = open(fName,'w', encoding="utf-8")
		content = bookSoup.get_text()
		start = content.find("Click here for ordering information.")
		end = content.find("Excerpted from")
		f.write(content[start+38:end])
		f.close()
		count += 1
