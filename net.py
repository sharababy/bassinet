import numpy as np;

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		return dict

imgset = unpickle("./cifar/data_batch_1");

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];

class NeuralNet:

	def trainThis(image,self):
		print(self[3])


net =[np.random.randint(0,1000,(3072,1),dtype=np.int64) * np.random.randint(0,1000,(1,3072),dtype=np.int64),
np.random.randint(0,1000,(10,1),dtype=np.int64) * np.random.randint(0,1000,(1,3072),dtype=np.int64),
np.random.randint(0,1000,(10,1),dtype=np.int64) * np.random.randint(0,1000,(1,10),dtype=np.int64),
np.random.randint(0,1000,(10,1),dtype=np.int64) * np.random.randint(0,1000,(1,10),dtype=np.int64)]





'''

# to print any label 
[print(classes[imgset[b'labels'][i]]) for i in range(10)]


[3072 x 3072][3072 x 1] = [3072 x 1]   -- layer 1

[10 x 3072][3072 x 1] = [10 x 1]       -- layer 2

[10 x 10][10 x 1] = [10 x 1]           -- layer 3

[10 x 10][10 x 1] = [10 x 1]           -- layer 4


'''


