'''
FizzBuzz Problem:

for each of the number 1 to 100:
	if number is divisible by 3:	print "fizz"
	if number is divisible by 5:	print "buzz"
	if number is divisible by 15:	print "fizzbuzz"
	otherwise print the number
'''
import numpy as np
from skynet.train import train
from skynet.nn import NeuralNet
from skynet.layers import Linear, Tanh

def fizz_buzz_encode(x:int)-> List[int]:
	if x%15==0:
		return [0,0,0,1]
	elif x%5==0:
		return[0,0,1,0]
	elif x%3==0:
		return[0,1,0,0]
	else:
		return[1,0,0,0]

def binary_encode(x:int)->List[int]:
	'''10 digit bin encoding of x'''
	return [x>> i &1 for i in range(10)]

inputs = np.array([
	binary_encode(x) for x in range(101,1024)
	])
targets = nparray([
	fizz_buzz_encode(x)
	for x in range(101,1024)
	])

net = NeuralNet([
	Linear(input_size = 10, output_size = 50 ),
	Tanh(),
	Linear(input_size=50,output_size=4)
	])

train(net, inputs, targets, num_epochs=5000)

for x in range(1,101):
	predicted = net.forward(binary_encode(x))
	predicted_index = np.argmax(predicted)
	actual_index = np.argmax(fizz_buzz_encode(x))
	labels = [str(x),"fizz", "buzz", "fizzbuzz"]
	print(x, labels[predicted_index], labels[actual_index])