import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from Modules.dropout import Dropout
from Modules.batchnorm import BatchNorm


class ConvNet(nn.Module):
    def __init__(self, params):

		#in the constructor we instanciate a list of modules that will contain each convolutional layer
		# we also instanciate a list of pooling layer
		#we assign them as member variables

        super(ConvNet, self).__init__()
        self.params = params
        self.Convs = nn.ModuleList()
        Conv = nn.Conv2d(3, self.params["channel_out"][0], self.params["kernel_size"][0], self.params["stride"][0], self.params["padding"][0]) #first convolutional layer
        self.Convs.append(Conv)
        if self.params["nb_conv_layers"] != 1:
            for i in range(1,self.params["nb_conv_layers"]):
                Conv = nn.Conv2d(self.params["channel_out"][i-1], self.params["channel_out"][i], self.params["kernel_size"][i], self.params["stride"][i], self.params["padding"][i])
                self.Convs.append(Conv)

        self.Pools = nn.ModuleList()

		#one pooling layer per convolution layer
        for i in range(self.params["nb_conv_layers"]):
            Pool = nn.MaxPool2d((self.params["pool_kernel_size"][i], self.params["pool_kernel_size"][i]))
			#-----------------pool_kernel_stride??????
            self.Pools.append(Pool)
            
        self.BatchNs = nn.ModuleList()

        nb_params = 3*params["channel_out"][0]*params["kernel_size"][0]*params["kernel_size"][0]
        for i in range(1, params["nb_conv_layers"]):
            nb_params += params["channel_out"][i]*params["channel_out"][i-1]*params["kernel_size"][i]*params["kernel_size"][i]
		#with convolution and pooling, it is not easy to control the size of an output
		#to create a final outlayer in 1D, it is easier to use a simple linear transformation (fully connected layer)
		#it is thus necessary to calculate the number of features we get after the last pooling
        W = self.params["W_in"]
        H = self.params["H_in"]
        for i in range(self.params["nb_conv_layers"]):
            W = out_dim_conv(W, self.params["padding"][i], 1, self.params["kernel_size"][i], self.params["stride"][i])
            H = out_dim_conv(H, self.params["padding"][i], 1, self.params["kernel_size"][i], self.params["stride"][i])
            BatchN = BatchNorm((self.params["channel_out"][i], H, W))
            self.BatchNs.append(BatchN)
            W = out_dim_pool(W, self.params["pool_kernel_size"][i])
            H = out_dim_pool(H, self.params["pool_kernel_size"][i])

        nb_params += H*W*self.params["channel_out"][-1]*self.params["lin_layers_size"] + self.params["lin_layers_size"]
        nb_params += self.params["lin_layers_size"]*self.params["lin_layers_size"] + self.params["lin_layers_size"]
        nb_params += self.params["lin_layers_size"]*2 + 2
        self.OutLayer1 = nn.Linear(H*W*self.params["channel_out"][-1], self.params["lin_layers_size"])
        self.OutLayer2 = nn.Linear(self.params["lin_layers_size"], self.params["lin_layers_size"])
        self.OutLayer3 = nn.Linear(self.params["lin_layers_size"], 2)
        #self.Dropout = Dropout(p=0.0)
        
        if self.params["modified_loss"]:
            self.W = nn.Linear(64*64,2)
            a = t.zeros(2)
            a.uniform_(-1, 1)
            self.a = nn.Parameter(a)
            self.a.requires_grad = True
        
        return


    def forward(self, input_batch):

		#the forward function compute the predicted class using operations on tensors
		#we use the modules defined in the construcor to return a tensor of output data

        batch_size = input_batch.size()[0]
        for i in range(self.params["nb_conv_layers"]):

			#we apply a Relu activation to each convolutional layer
            input_batch = F.relu(self.Convs[i](input_batch))
            input_batch = self.Pools[i](input_batch)
            #input_batch = self.Dropout(input_batch)
        output_preac_batch = self.OutLayer3(F.relu(self.OutLayer2(F.relu(self.OutLayer1(input_batch.contiguous().view(batch_size, -1))))))
        if self.training and self.params["modified_loss"]:
            output_preac_batch = output_preac_batch + self.a*t.exp(self.W(input_batch.view(-1, 64*64)))
        return output_preac_batch


    def trainer(self, optimizer):
        def train_step(batch, use_cuda=False):

			#we must set the mode to train (some layers can behave diffrent on the train and test procedures)
            self.train()
            lambda_ = 1

			#as the forward function computes output Tensors from input Tensors,we should transform numpy arrays to PyTorch tensors
			#Tensors can keep track of a computational graph and gradients
			#(each Tensor represents a node in a computational graph)
			#PyTorch Tensors can utilize GPUs to accelerate their numeric computations(Numpy can't)
            input_batch = batch[0]
            target_batch = batch[1]
            if use_cuda:
                input_batch = t.from_numpy(input_batch).float().cuda()
                target_batch = t.from_numpy(target_batch).long().cuda()
            else:
                input_batch = t.from_numpy(input_batch).float()
                target_batch = t.from_numpy(target_batch).long()

			#the gradients of all optimized torch.Tensors must be cleared before getting new values
            optimizer.zero_grad()

			#final out preactivation returned by Convnet.forward function
            output_preac_batch = self(input_batch)

			#the loss function is the softmax function used in tandem with the negative log-likelihood (NLL)
            loss = F.cross_entropy(output_preac_batch, target_batch)
            if self.params["modified_loss"]:
                loss = loss + lambda_*t.norm(self.a)**2

			#calculating gradients
            loss.backward()

			#updating the parameters thranks to the gradients
            optimizer.step()
            
            pred = [int(np.argmax(output_preac)) for output_preac in output_preac_batch.data.cpu().numpy()]
            acc = sum([1 if target_batch[i] == pred[i] else 0 for i in range(len(pred))])/len(pred)

			#to get the loss data we should move it to cpu
            return float(loss.data.cpu().numpy()), acc
        return train_step


    def validator(self):
        def valid_step(batch, use_cuda=False):
            self.eval()
            input_batch = batch[0]
            target_batch = batch[1]
            if use_cuda:
                input_batch = t.from_numpy(input_batch).float().cuda()
                target_batch = t.from_numpy(target_batch).long().cuda()
            else:
                input_batch = t.from_numpy(input_batch).float()
                target_batch = t.from_numpy(target_batch).long()
            output_preac_batch = self(input_batch)
            loss = F.cross_entropy(output_preac_batch, target_batch)
            pred = [int(np.argmax(output_preac)) for output_preac in output_preac_batch.data.cpu().numpy()]
            acc = sum([1 if target_batch[i] == pred[i] else 0 for i in range(len(pred))])/len(pred)


            return float(loss.data.cpu().numpy()), acc
        return valid_step


    def tester(self):
        def test_step(batch, use_cuda=False):
            self.eval()
            input_batch = batch
            if use_cuda:
                input_batch = t.from_numpy(input_batch).float().cuda()
            else:
                input_batch = t.from_numpy(input_batch).float()
            output_preac_batch = self(input_batch)
            pred = [int(np.argmax(output_preac)) for output_preac in output_preac_batch.data.cpu().numpy()]

            return pred
        return test_step


def out_dim_conv(in_dim, padding, dilation, kernel_size, stride):
    return math.floor((in_dim + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)

def out_dim_pool(in_dim, kernel_size):
    return math.floor((in_dim - (kernel_size - 1) - 1)/kernel_size + 1)
