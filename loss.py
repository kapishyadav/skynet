'''
loss function is used to measure the accuracy of the model.
Here I'll be implementing the mean squared error loss function.
'''
import numpy as np
from skynet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
	def loss(self, predicted:Tensor,actual:Tensor)-> float:
		return np.sum((predicted - actual)**2)

	def grad(self, predicted:Tensor,actual:Tensor)->Tensor:   # Since the gradient is the partial derivative of the loss function
		return 2*(predicted- actual)
		
	


