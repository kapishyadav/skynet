'''
Optimizers are used to reduce the loss function
by adjusting the parameters of out neural network 
based on the gradient computed during backpropagation
'''
from skynet.nn import NeuralNet
class Optimizer:
	def step(self, net: NeuralNet)->None:
		raise NotImplementedError


class SGD(Optimizer):
	def __init__(self, lr: float= 0.01)->None:
		self.lr=lr

	def step(self, net:NeuralNet)->None:
		for param, grad in net.praram_and_grads():
			param-=self.lr*grad

