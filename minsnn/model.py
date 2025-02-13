import torch.nn as nn 
import torch
import numpy as np 
from neurons import Leaky
# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
    def __init__(self, beta, threshold=1.0):
        super(LeakySurrogate, self).__init__()

        self.beta = beta 
        self.threshold = threshold
        self.spike_gradient = self.ATan.apply 

        self.mem = None
        
    def forward(self, input_, mem=None):
        if mem is None:
            self.mem = torch.zeros_like(input_)  # Initialize with correct shape
        else:
            self.mem = mem  # Use the given mem state

        spk = self.spike_gradient((mem - self.threshold))
        reset = (self.beta * spk * self.threshold).detach() 
        # print(f'mem.shape:{mem.shape}')
        # print(F'input_.shape:{input_.shape}')
        # exit()
        mem = self.beta * mem + input_ - reset 
        return spk, mem 
    def init_leaky(self, input_shape):
        return torch.zeros(input_shape, dtype=torch.float)

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float() 
            ctx.save_for_backward(mem) 
            return spk 
        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors 
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output
            return grad
# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = LeakySurrogate(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = LeakySurrogate(beta=beta)

    def forward(self, x):
        
        batch_size = x.shape[0]
        hidden_shape = (batch_size, self.fc1.out_features)
        output_shape = (batch_size, self.fc2.out_features)



        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky(hidden_shape)
        mem2 = self.lif2.init_leaky(output_shape)

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            # print(f'--Step:{step}--')
            # print(f'x.shape:{x.shape}')
            # print(f'mem1.shape:{mem1.shape}')
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)        
