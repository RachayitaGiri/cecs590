
train = open("train.txt", "w")
test = open("test.txt", "w")
vali = open("validate.txt", "w")
file3 = open("words.txt", "r")
keys = []
for i in range(1,70000):
    temp = str(file3.readline())
    keys.append(temp[0:9])

stuff = []
file = open("fall11_urls.txt", "r")
count = 0
pastID = ""
total = 0
inFile = 0
arr = []
size = 100
for i in range(0, 14195091):
    
    temp = str(file.readline())
    catId = temp[0:9]
    if(catId != pastID):
        if(count >= size):
            print(str(total) + ": " + catId + ": " + str(count) + " InFile: " + str(inFile))
            inList = 0
            while(inList < len(arr)):
                if(inList < len(arr) * 0.6):
                    train.write(arr[inList])
                elif(inList < len(arr) * 0.8):
                    test.write(arr[inList])
                elif(inList < len(arr) * 0.8):
                    vali.write(arr[inList])
                inList = inList + 1

            
            total = total + 1
            inFile = 0

        arr = []
        count = 0
        inFile = 0
    if(count < size):
        arr.append(temp)
        #file2.write(temp)
        inFile = inFile + 1
    pastID = catId
    count = count + 1
    
    
    #t = str(file.readline())
    #stuff.append(t)



# for i in range(len(keys)):
    
#     count = 0
    
#     for j in range(len(stuff)):
        
#         if count == 50:
#             break
#         if(keys[i] == str(stuff[j][0:9])):
#             file2.write(stuff[i])
#             count = count + 1
#     print(str(i) + ": " +str(keys[i]) + ": " + str(count))
    
#print(keys[0])

file2.close()
