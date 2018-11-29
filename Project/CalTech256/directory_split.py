# This code splits the source directory into test and validation sets.
# Whatever remains in the original directory is the training datatset.
import os
import shutil
import numpy as np

srcDirName = "256_ObjectCategories"

#uncomment the next line if running for test folder
#dstDirName = "test"

#uncomment the next line if running for validation folder
#dstDirName = "validation"

files = os.listdir(srcDirName)

for f in files:
    if np.random.rand(1) < 0.2:
        shutil.move(srcDirName + '/'+ f, dstDirName + '/'+ f)