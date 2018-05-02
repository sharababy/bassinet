import numpy as np;

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		return dict

imgset = unpickle("./cifar/data_batch_1");

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];

class NeuralNet:

	def __init__(self):
		self.layer = [np.random.ranf((3072,1)) * np.random.ranf((1,3072)),
		np.random.ranf((10,1)) * np.random.ranf((1,3072)),
		np.random.ranf((10,1)) * np.random.ranf((1,10)),
		np.random.ranf((10,1)) * np.random.ranf((1,10))]	

	def trainThis(self,image,cnumber):
		out1 = np.matmul(self.layer[0] , image)

		out1 = [x/10000 for x in out1]
		out2 = np.matmul(self.layer[1] , out1)
		out3 = np.matmul(self.layer[2] , out2)
		out4 = np.matmul(self.layer[3] , out3)

		# print(out1)
		# print(out2)
		# print(out3)
		# print(out4)

		errors = 

		return out4;

	def printLayer(self,n):
		print(self.layer[n])


net = NeuralNet();

flayer = net.trainThis(imgset[b'data'][0]);



# print(flayer)


'''

# to print any label 
[print(classes[imgset[b'labels'][i]]) for i in range(10)]

# tp print image data
# print(imgset[b'data'])

[3072 x 3072][3072 x 1] = [3072 x 1]   -- layer 1

[10 x 3072][3072 x 1] = [10 x 1]       -- layer 2

[10 x 10][10 x 1] = [10 x 1]           -- layer 3

[10 x 10][10 x 1] = [10 x 1]           -- layer 4


'''


