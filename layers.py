'''
Neural Networks contains layers which are use to propagate through the network.
Each layer needs to pass it's inputs forward and propagate gradients backwards
Eg:
	input -> Linear_func -> tanh_func -> Linear_func -> output

'''
from typing import Dict, Callable
import numpy as np
from skynet.tensor import Tensor

class Layer:

	def __init__(self,):
		self.params: Dict[str, Tensor] = {}
		self.grads: Dict[str, Tensor] = {}

	def forward(self, inputs: Tensor, outputs: Tensor):
		raise NotImplementedError

	def backward(self, inputs: Tensor, outputs: Tensor):
		raise NotImplementedError


class Linear(Layer):
	'''
	Calculates output = inputs * weights + biases
	'''
	def __init__(self):
		'''
		inputs size: (batch_size , input_size)
		output size: (batch_size , output_size)
		'''

		super().__init__()
		self.params["w"] = np.random.randn(input_size, output_size)
		self.param["b"] = np.random.randn(output_size)

	def forward(self, inputs: Tensor, outputs: Tensor) -> Tensor:
		
		'''
		Forward pass Output =  inputs * weights + biases
		'''
		self.inputs = inputs
		return inputs @ self.params["w"] + self.params["b"]

	def backward(self, inputs: Tensor, outputs: Tensor) -> Tensor:
		'''
		Backward pass is Backpropagation where:

		if y = f(x) and x = a*b+c
		then dy/da = f'(x)*b
			 dy/db = f'(x)*a
			 dy/dc = f'(x)
		
		if y = f(x) and x = a@b+c
		then dy/da = f'(x)@b.T
			 dy/db = a.T@f'(x)
			 dy/dc = f'(x)
		
		'''
		self.grads["b"] = np.sum(grad, axis=0)
		self.grads["w"] = self.inputs.T@grad
		return grad@self.params["w"].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):

	def __init__(self, f: F, f_prime: F):
		self.f = f
		self.f_prime = f_prime
		
	def forward(self, inputs: Tensor, outputs: Tensor):
		'''
		applies activation function to all inputs
		'''
		self.inputs = inputs
		return self.f(inputs)

	def backward(self, inputs: Tensor, outputs: Tensor):
		'''
		if y = f(x) and f(x) = g(z)
		then dy/dz = f'(x)*g'(x)
		'''
		return grad*f_prime(self.inputs)


def tanh(x:Tensor)->Tensor:
	return np.tanh(x)

def tanh_prime(x:Tensor)->Tensor:
	y=tanh(x)
	return 1-y**2

class Tanh(Activation):
	def __init__(self):
		super().__init__(tanh, tanh_prime)
