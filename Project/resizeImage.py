import os, sys

import PIL
from PIL import Image
from resizeimage import resizeimage

def resize(imageName):
    try:
        basewidth = 224
        image = imageName
        img = Image.open(image)
        img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
        img.save(imageName)
        #os.rename(imageName, str(newName) + ".jpeg") 
    except:
        os.remove(imageName)

def main():
    # Open a file
    dirName = "trainSet"

    os.chdir(dirName)
    dirs = os.listdir(os.getcwd())

    for file in dirs:
        if(file != ".DS_Store"): 
            os.chdir(file)
            allPictures = os.listdir(os.getcwd())
            
            for image in allPictures:
                print(image)
                resize(image)
            allPictures = os.listdir(os.getcwd())
            count = 1
            for image in allPictures:
                os.rename(image, str(count) + ".jpeg")
                count = count + 1 


            os.chdir("..")
        else:
            os.remove(".DS_Store")

main()
