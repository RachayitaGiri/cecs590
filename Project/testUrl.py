import requests
import urllib.request
import os


def downloader(image_url, file_name):
    # try:
    #     print(image_url)
    #     urllib.request.urlretrieve(image_url,file_name)
    #     size = os.path.getsize(file_name)
    #     if(size < 2500):
    #         os.remove(file_name)
    # except:
    #     pass

    try:
        print(image_url)
        r = requests.get(image_url, timeout = 5).content
        f = open(file_name,'wb')
        f.write(r)
        f.close()
        size = os.path.getsize(file_name)
        print(size)
        if(size < 4000):
            os.remove(file_name)  
    except:
        pass

dirName = 'testSet'
file = open("test.txt", "r")

try:
    # Create target Directory
    os.mkdir(dirName)
    #os.chdir(dirName)
    
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

os.chdir(dirName)

pastName = dirName
for i in range(10000000):
    
    temp = file.readline()
    index = temp.find("http")
    name = temp[:index].replace(" ","") 
    nameDir = name[:name.find("_")]
    if(nameDir != pastName):
        if(pastName != dirName):
            os.chdir("..")
        try:
            # Create target Directory
            os.mkdir(nameDir)
            os.chdir(nameDir)
        except FileExistsError:
            #print("Directory " , dirName ,  " already exists")
            os.chdir(nameDir)
    pastName = nameDir


    result = temp[index:].replace("\n", "")
    name = temp[:index] + ".jpeg"

    downloader(result, name)
    
    # try:
    #     r = requests.get(result).content
    #     f = open(name,'wb')
    #     f.write(r)
    #     f.close()
    # except:
    #     pass
    
    


