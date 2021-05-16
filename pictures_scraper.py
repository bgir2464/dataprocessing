import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests

from parse_csv import *


URL = 'https://archaeologydataservice.ac.uk/archives/view/alanvince_eh_2010/downloads.cfm?archive=other'
BASE_URL = 'https://archaeologydataservice.ac.uk/'

option = {'archives': 'Northern_England_Medieval_Whitewares'}
page = requests.post(URL, option)

DEBUG=1
def download(url, pathname):
    """
    Downloads a file given an URL and puts it in the folder `pathname`
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # get the file name
    filename = os.path.join(pathname, url.split("/")[-1])
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True,
                    unit_divisor=1024)

    with open(filename, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))


def download_tdar_2012_byland():
    url = 'https://core.tdar.org/search/results?dataMappedCollectionId=22070&groups%5B0%5D.dataValues%5B0%5D.columnId=74159&groups%5B0%5D.dataValues%5B0%5D.singleToken=false&objectTypes=IMAGE&startRecord=2150&id=&keywordType=&slug=&recordsPerPage=25&orientation=GRID'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    res = soup.find("div", {"class": "tdarresults"})
    print(res)


def download_file(f,folder):
    headers={
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
        "cookie":"_ga=GA1.2.188017350.1614358174; _gid=GA1.2.38117591.1617136515; JSESSIONID=2FD5636C6B432233B22DD94BD84688CB; crowd.token_key=72-vtkxe6D13qr9ph9oYzwAAAAAAAAABZ2VvcmdpYWJ1Y2Vh; _gat=1",
        "authority":"core.tdar.org"
    }
    response = requests.get(f,headers=headers)

    # print(response.headers)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    file_size = int(response.headers.get("Content-Length", 0))
    path=f.split("/")
    # print(path)
    #
    filename = os.path.join(folder,path[-1]+"_"+path[-2]+".tiff")
    if not os.path.isfile(filename):
        print(filename)
        progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True,unit_divisor=1024)
        with open(filename, "wb") as f:
            for data in progress:
                # write data read to the file
                f.write(data)
                # update the progress bar manually
                progress.update(len(data))


def download_tdar_2012_byland_picture(URL,folder):
    try:
        page = requests.get(URL)
        base_url="https://core.tdar.org"
        soup = BeautifulSoup(page.content, 'html.parser')
        res = soup.find("div", {"id": "extendedFileInfoContainer"})
        # print(URL)
        images_download=res.find_all('a',{"class":"download-link download-file"})
        for image in images_download:
            image=image['href']
            # print(image)
            path=image.split("/")
            image_download_path=base_url+"/"+path[1]+"/"+"get"+"/"+path[3]+"/"+path[4]
            download_file(image_download_path,folder)
        # img=requests.get(image_download_path,payload)
        # print(img.text)
        #download(image_download_path,"tdar_2012")
    except Exception as e:
        print("download_tdar_2012_byland_picture: "+str(e))

def downloads_alan():
    soup = BeautifulSoup(page.content, 'html.parser')
    tabels = soup.find("div", {"id": "archive"}).find_all("table", {"class": "dltab"})
    for el in tabels:
        links = el.find_all('a')
        for link in links:
            if link.get('href'):
                if link.get('href').find('jpg') != -1:
                    lk = link.get('href')
                    # li[-2] has the folder name
                    li = lk.split('/')
                    print(li)
                    picture_url = BASE_URL + link.get('href')
                    download(picture_url, li[-2])


# download_tdar_2012_byland_picture("https://core.tdar.org/image/382392/10020-style-iii-bowl-from-swarts")
#
# f="https://core.tdar.org/filestore/get/381956/213913"
# download_file(f,"tdat_2012")

file_name = 'report_2.xlsx' # change it to the name of your excel file
urls=get_pictures_urls(file_name)

urls_with=get_pictures_urls("report_with.xlsx")
folder_with="report_2012_with"
folder_without="report_2012_without"
urls_without=get_pictures_urls("report_without.xlsx")

DEBUG=1
if DEBUG == 0:
    for url in urls_without:
        download_tdar_2012_byland_picture(url,folder_without)
if DEBUG ==1:
    download_tdar_2012_byland_picture("https://core.tdar.org/image/381928/124-style-iii-bowl-from-manuel-gonzales")

