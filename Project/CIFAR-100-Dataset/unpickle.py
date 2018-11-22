import json

# unpickle the given file and return a dictionary
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# call to the unpickle() function
train_data = unpickle("train")
test_data = unpickle("test")
meta_data = unpickle("meta")

# convert the binary key/value pairs to string key/value pairs
train_data = dict({str(k):str(v) for k, v in train_data.items()})
test_data = dict({str(k):str(v) for k, v in test_data.items()})
meta_data = dict({str(k):str(v) for k, v in meta_data.items()})

# write the dictionaries in a json file
with open('train_file.json', 'w') as file:
     file.write(json.dumps(train_data, sort_keys=True, indent=4, separators=(',', ': ')))

with open('test_file.json', 'w') as file:
     file.write(json.dumps(test_data, sort_keys=True, indent=4, separators=(',', ': ')))

with open('meta_file.json', 'w') as file:
     file.write(json.dumps(meta_data, sort_keys=True, indent=4, separators=(',', ': ')))
