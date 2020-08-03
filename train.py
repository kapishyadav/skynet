'''
A function that can train a neural net
'''

from tensor import Tensor
from nn import NeuralNet
from optim import SGD, Optimizer
from loss import Loss, MSE
from data import DataIterator, BatchIterator 

def train(net: NeuralNet,
	inputs: Tensor,
	targets: Tensor,
	num_epochs: int=5000,
	iterator: DataIterator = BatchIterator(),
	loss: Loss=MSE(),
	optimizer: Optimizer=SGD())->None:
	
	for epoch in range(num_epochs):
		epoch_loss = 0.0
		for batch in iterator(inputs, targets):
			predicted = net.forward(batch.inputs)
			epoch_loss+=loss.loss(predicted, batch.targets)
			grad=loss.grad(predicted,batch.targets)
			net.backward(grad)
			optimizer.step(net)
		print(epoch, epoch_loss)
	print("Final loss: ",epoch_loss)


