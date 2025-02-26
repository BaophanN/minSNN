import torch.nn as nn 
import torch
import snntorch as snn 
import numpy as np 
from neurons import Leaky
from tqdm import tqdm 
from torch.optim import Adam 
import torch.nn.functional as F 

device = "cuda:0"
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

class LeakyLayer(nn.Linear): 
    def __init__(self, in_features, out_features, activation, bias=False):
        super().__init__(in_features, out_features, bias=bias) 

        # choose spiking neuron or standard neuron 
        if activation == "lif":
            self.activation = snn.Leaky(beta=0.8) 
            self.lif = True 
        else:
            self.activation = nn.ReLU() 
            self.lif = False 
        
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 5.0 
        self.num_epochs = 1000 

    def forward(self, x): 
        if self.lif == True: 
            mem = self.activation.init_leaky() # init membrane potential 

        x_direction = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-4) # normalize the input 

        # Linear layer - not using nn.Linear(...) so we can define the update rule
        weighted_input = torch.mm(x_direction, self.weight.T.to(device)) 
        # Matmul between normalized input and weight matrix

        if self.lif == True: 
            spk, potential = self.activation(weighted_input, mem) # note: only 1 step in time. Wrap in a for-loop to iterate for longer 
        else: 
            potential = self.activation(weighted_input) 
        
        return potential # to be used in subsequent layers or as the final output of the last layer 

    def train(self, x_pos, x_neg): 
        tot_loss = [] # store the loss values for each layer in each epoch 
        for _ in tqdm(range(self.num_epochs), desc="Training LeakyLayer"):
            # Compute goodness 
            g_pos = self.forward(x_pos).pow(2).mean(1) # positive data 
            g_neg = self.forward(x_neg).pow(2).mean(1) # negative data 

            # take the mean of differences between goodness and threshold across pos and neg 
            loss = F.softplus(torch.cat([-g_pos + self.threshold, g_neg - self.threshold])).mean() 

            self.opt.zero_grad() 
            loss.backward() # local backward-pass 
            self.opt.step() # update weights 
            tot_loss.append(loss) 
        
        # returns the final membran potentials (activations) for positive and negaive examples after training. 
        # detach() ensures no further backward pass is possible 
        output = self.forward(x_pos).detach(), self.forward(x_neg).detach()
        return (output, tot_loss)

class FF_Net(nn.Module):
    def __init__(self, dims, activation): 
        super().__init__()

        self.layers = nn.ModuleList([
            LeakyLayer(dims[d], dims[d+1], activation) for d in range(len(dims) - 1) # define a multi-layer network 
        ]) 

            
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
