import pprint
from tempfile import mktemp
from urllib.request import urlopen, Request, urlretrieve
from zipfile import ZipFile

from bs4 import BeautifulSoup
import requests

URL = 'https://archaeologydataservice.ac.uk/archives/view/hayton_eh_2007/'


URL='https://archaeologydataservice.ac.uk/archives/view/alanvince_eh_2010/'

page = requests.get(URL)
pp = pprint.PrettyPrinter(indent=4)

soup = BeautifulSoup(page.content,'html.parser')

metadata=soup.find( id='metadata').find('ul','dlMenu').find_all('li')
downloads_link=None
for el in metadata:
    if el.find('a').get('href').find('downloads')!=-1:
       downloads_link=el.find('a').get('href')
if downloads_link!=None:
    downloads_url=URL+downloads_link


page_downloads = requests.get(downloads_url).content


soup = BeautifulSoup(page_downloads,'html.parser')
archive=soup.find("div", {"id": "archive"})

links = archive.find_all('a')

for el in links:
    # if el['href'].find('form')!=-1:
    page = requests.get(URL+el['href'])
    soup = BeautifulSoup(page.content, 'html.parser')
    links2 = soup.find_all('a')
    for link in links2:
            if link.has_attr('href'):
                href=link['href']
                if '.zip' in href:
                    # remoteZip = urlopen(Request(URL+href))
                    file_name = href.rpartition('/')[-1]
                    file=file_name.split('?')
                    filename = file[0]

                    destDir = mktemp(filename.split(".")[0])
                    theurl = URL+href
                    name, hdrs = urlretrieve(theurl, filename)
                    thefile = ZipFile(filename)
                    thefile.extractall(destDir)
                    thefile.close()

                    # file_name = href.rpartition('/')[-1]
                    # file=file_name.split('?')
                    # local_file = open(file_name[0], 'wb+')
                    # local_file.write(remoteZip.read())
                    # local_file.close()










