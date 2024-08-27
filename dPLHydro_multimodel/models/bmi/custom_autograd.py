"""
A custom backward method to enable gradient backpropogation in a
physics-informed, differentiable ML BMI.
"""
import torch
import numpy as np
from typing import Optional
from torch.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge



class BMIBackward(torch.autograd.Function):
    """
    Custom autograd with torch.autograd.Function for BMI-friendly backward pass 
    implementation which operate on Tensors.

    Essentially, we need this to calculate dL/dw = del(L)/del(y) * dy/dw,
    then torch Autodiff system can do the rest (e.g., calc vector Jacobian prod).
    """

    # @staticmethod
    # def forward(ctx, input):
    #     """
    #     In the forward pass we receive a Tensor containing the input and return
    #     a Tensor containing the output. ctx is a context object that can be used
    #     to stash information for backward computation. You can cache arbitrary
    #     objects for use in the backward pass using the ctx.save_for_backward method.
    #     """
    #     ctx.save_for_backward(input)
    #     return input.clamp(min=0)

    @staticmethod
    def backward(tensors: _TensorOrTensors,
                 grad_tensors: Optional[_TensorOrTensors] = None,
                 retain_graph: Optional[bool] = None,
                 create_graph: bool = False,
                 grad_variables: Optional[_TensorOrTensors] = None,
                 inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
                 ) -> None:
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    