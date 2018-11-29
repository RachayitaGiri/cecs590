# This code gets rid of the numerical prefixes of the folder names.
# e.g '013.helicopter' --> 'helicopter'
import os

dirName = "256_ObjectCategories/"
dirs = os.listdir(dirName)
try:
	for directory in sorted(dirs):
		newid, newname = directory.split(".")
		if directory==".DS_Store":
			pass
		else:
			print(directory)
			os.rename(dirName+directory+"/", dirName+newname+"/")

except Exception as e:
	print(e)