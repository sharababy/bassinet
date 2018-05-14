import numpy as np;

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		return dict

imgset = unpickle("./cifar/data_batch_1");

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];

classcount = 10

p = 0

class NeuralNet:

	def __init__(self):
		self.layer = [np.random.ranf((3072,1)) * np.random.ranf((1,3072)),
		np.random.ranf((10,1)) * np.random.ranf((1,3072)),
		np.random.ranf((10,1)) * np.random.ranf((1,10)),
		np.random.ranf((10,1)) * np.random.ranf((1,10))]	

	def predictThis(self,image,c,correct):
		out0 = np.matmul(self.layer[0] , image)
		maxout0 = np.amax(out0)
		out0 = [x/(maxout0+10) for x in out0]
		
		pout1 = np.matmul(self.layer[1] , out0)
		maxout1 = np.amax(pout1)
		pout1 = [x/(maxout1+10) for x in pout1]

		out2 = np.matmul(self.layer[2] , pout1)
		maxout2 = np.amax(out2)
		out2 = [x/(maxout2+10) for x in out2]

		out2 = np.array(out2).ravel()
		

		out3 = np.matmul(self.layer[3] , out2)
		maxout3 = np.amax(out3)
		out3 = [x/(maxout3+10) for x in out3]

		# print(" predicted = ",np.argmax(out3)," class = ",c)

		if(np.argmax(out3) == c):
			correct+=1
		
		return correct

	def trainThis(self,image,c):
		out0 = np.matmul(self.layer[0] , image)
		maxout0 = np.amax(out0)
		out0 = [x/(maxout0+10) for x in out0]
		
		out1 = np.matmul(self.layer[1] , out0)
		maxout1 = np.amax(out1)
		out1 = [x/(maxout1+10) for x in out1]

		out2 = np.matmul(self.layer[2] , out1)
		maxout2 = np.amax(out2)
		out2 = [x/(maxout2+10) for x in out2]

		out2 = np.array(out2).ravel()
		out3 = np.matmul(self.layer[3] ,out2)
		maxout3 = np.amax(out3)
		out3 = [x/(maxout3+10) for x in out3]
		

		ideal = np.zeros(10)
		ideal.fill(0.1)
		ideal[c] = 0.9

		# print(" ideal: ",ideal)
		# print(" error: ",error)
		# print("Layer 4","\t\t\t","Error")
		# [ print(out3[i],"\t\t",error[i]) for i in range(10) ]
		
		# backprop for layer 3
		layer3ideal = np.matmul(np.asmatrix(ideal).transpose(),np.asmatrix(out2))
		l3diff =  layer3ideal - self.layer[3]
		self.layer[3] = self.layer[3] + (l3diff/5)

		
		global p
		p = net.predictThis(image,c,p)

		return out3;

	def printLayer(self,n):
		print(self.layer[n])


net = NeuralNet();

trainCount = 1800

[net.trainThis(imgset[b'data'][i],imgset[b'labels'][i]) for i in range(trainCount)]

print("--------------------------------------------")
print("Now for the tests: ")

print(p)

correct = 0
total = 50

for i in range(2001,2001+total):
	correct = net.predictThis(imgset[b'data'][i],imgset[b'labels'][i],correct)

print("\n\nAccuracy: ",correct,"/",total)


# print(flayer)


'''

# to print any label 
[print(classes[imgset[b'labels'][i]]) for i in range(10)]

# tp print image data
# print(imgset[b'data'])

[3072 x 3072][3072 x 1] = [3072 x 1]   -- layer 0

[10 x 3072][3072 x 1] = [10 x 1]       -- layer 1

[3072 x 10][10 x 1] = [3072 x 1]

[10 x 10][10 x 1] = [10 x 1]           -- layer 2
[10 x 10] = [10 x 1] [10 x 1].T

[10 x 10][10 x 1] = [10 x 1]           -- layer 3
[10 x 10] = [10 x 1] [1 x 10]t 

'''


