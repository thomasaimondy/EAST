# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

 "models.py" - Construction of arbitrary network topologies.

 Project: PM - Predictive Modification  

 Authors: liuhongxing  02/2021

 基于 BRP，在每一个block中增加了下面的代码：
 if self.dim_hook is not None and labels is not None:
    # 隐层输出梯度
    grad_output = labels.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape)  
    # grad_output.zero_() # gradient set 0, donot modify the weights
    # 更新局部梯度
    self.loacl_gradient_pytorch(self.spike, grad_output) # 封装的局部梯度更新
使得网络的隐层在前传的过程中可利用 B*Label 更新当前层的权重梯度，并且对隐层的输出使用了.detach() 截断层间的梯度计算

------------------------------------------------------------------------------
"""

import torch
import numpy as np
from numpy import prod
import torch.nn as nn
import torch.optim as optim
import function
from module import FA_wrapper, TrainingHook
import os
# thresh = 0.5
# randKill = 0.1
# lens = 0.5
# decay = 0.2
spike_args = {}
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
use_gpu_num = 1  # torch.cuda.device_count()

class NetworkBuilder(nn.Module):
    """
    This version of the network builder assumes stride-2 pooling operations.
    """

    def __init__(self, topology, input_size, input_channels, label_features, train_batch_size, train_mode, dropout,
                 conv_act, hidden_act, output_act, fc_zero_init, spike_window, device, thresh, randKill, lens, decay):
        super(NetworkBuilder, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_size = train_batch_size
        self.spike_window = spike_window
        self.randKill = randKill
        spike_args['thresh'] = thresh
        spike_args['lens'] = lens
        spike_args['decay'] = decay

        if (train_mode == "DFA") or (train_mode == "sDFA"):
            self.y = torch.zeros(train_batch_size, label_features, device=device)
            self.y.requires_grad = False
        else:
            self.y = None

        topology = topology.split('_')
        self.topology = topology
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not any(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers - 1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV":
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(
                        output_dim / 2)  # /2 accounts for pooling operation of the previous convolutional layer
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(CNN_block(
                        in_channels=in_channels,
                        out_channels=int(layer[1]),
                        kernel_size=int(layer[2]),
                        stride=int(layer[3]),
                        padding=int(layer[4]),
                        bias=True,
                        activation=conv_act,
                        dim_hook=[label_features, out_channels, output_dim, output_dim],
                        label_features=label_features,
                        train_mode=train_mode,
                        batch_size=self.batch_size,
                        spike_window=self.spike_window
                    ))
                elif layer[0] == "FC":
                    if (i == 0):
                        input_dim = pow(input_size,2)*input_channels
                        #input_dim = input_size
                        self.conv_to_fc = 0
                        # print('i=0')
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])  # /2 accounts for pooling operation of the previous convolutional layer
                        self.conv_to_fc = i
                        # print('conv')
                    elif topology_layers[i - 1][0] == "C":
                        input_dim = 1000  # /2 accounts for pooling operation of the previous convolutional layer
                        # input_dim = int(output_dim)
                        # print(input_dim)
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim
                        # print('else')

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(FC_block(
                        in_features=input_dim,
                        out_features=output_dim,
                        bias=True,
                        activation=output_act if output_layer else hidden_act,
                        dropout=dropout,
                        dim_hook=None if output_layer else [label_features, output_dim],
                        label_features=label_features,
                        fc_zero_init=fc_zero_init,
                        train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode,
                        batch_size=train_batch_size,
                        spike_window=self.spike_window
                    ))

                elif layer[0] == "C":
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(output_dim / 2)  # /2 accounts for pooling operation of the previous convolutional layer
                    output_dim = int((input_dim + 2*int(layer[4]) - int(layer[2]) + 1) / int(layer[3]))#维度对不上的话可以直接赋值，要不太难调了
                    self.layers.append(C_block(
                        in_channels=in_channels,
                        out_channels=int(layer[1]),
                        kernel_size=int(layer[2]),
                        stride=int(layer[3]),
                        padding=int(layer[4]),
                        bias=True,
                        activation=conv_act,
                        dim_hook=[label_features, out_channels, output_dim],
                        label_features=label_features,
                        train_mode=train_mode,
                        batch_size=self.batch_size,
                        spike_window=self.spike_window
                    ))
                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))

    def forward(self, input, labels, args, optimizer=None, batch_idx=-1):
        input = input.float().cuda()
        dataset = args.dataset

        for step in range(self.spike_window):
            if self.topology[0] == 'C':
                x = input[:, :, :, step]
            else:
                if dataset == 'MNIST' or dataset == 'CIFAR10' or dataset == 'HWDB' or dataset=='tidigits':
                    x = input > torch.rand(input.size()).float().cuda() * self.randKill
                    #print('x.shape:',x.shape)
                    #print(len(np.where(x[0,0,0,:].cpu()>0.5)[0]))
                if dataset == 'dvsgesture':
                    x = input[:, :, step, :, :] > torch.rand(input[:, :, 0, :, :].size()).float().cuda() * self.randKill
                #如果是mnist则x = input > torch.rand(input.size()).float().cuda() * self.randKill，下面的是dvsgesture的程序

            x = x.float()
            # print(x.max())

            for i in range(len(self.layers)):
                if i == self.conv_to_fc:
                    x = x.reshape(x.size(0), -1)
                x = self.layers[i](x, labels, self.y,optimizer,args,batch_idx).detach() # 截断隐层输出的梯度计算传递

        x = self.layers[-1].sumspike / self.spike_window  # 网络输出正常BP更新，因此保留梯度计算

        # if x.requires_grad and (self.y is not None):
        #     self.y.data.copy_(x.data)  # in-place update, only happens with (s)DFA

        return x


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(spike_args['thresh']).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - spike_args['thresh']) < spike_args['lens']
        return grad_input * temp.float()

    # @staticmethod  
    # def backward(ctx, grad_h):
    #     z = ctx.saved_tensors
    #     s = torch.sigmoid(z[0])
    #     d_input = (1 - s) * s * grad_h
    #     return d_input


act_fun = ActFun.apply


def mem_update(ops, x, mem, spike, args,lateral=None):
    a = torch.sigmoid(ops(x))
    # mem_mask = torch.ones_like(mem)
    # proportion = args.proportion
    # mem_mask[:,:int(proportion*mem.shape[1])] = mem[:,:int(proportion*mem.shape[1])]
    mem = mem * spike_args['decay'] * (1. - spike) + a # 看一下的输出是不是一直增大torch.sigmoid(ops(x))
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, label_features, fc_zero_init,
                 train_mode, batch_size, spike_window):
        super(FC_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.spike_window = spike_window
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.count = 0
        self.n = 0
        self.weight = []
        self.h = []

        self.dim_hook=dim_hook
        if dim_hook is not None:
            self.B = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def forward(self, x, labels, y,optimizer,args,batch_idx):
        # if self.dropout != 0:

        if self.time_counter == 0:
            self.mem = torch.zeros((int(self.batch_size/use_gpu_num), self.out_features)).cuda()
            self.spike = torch.zeros((int(self.batch_size/use_gpu_num), self.out_features)).cuda()
            self.sumspike = torch.zeros((int(self.batch_size/use_gpu_num), self.out_features)).cuda()

        if False:
            x = self.drop(x)

        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike, args)
        self.sumspike += self.spike

        # optimizer = optim.Adam(self.fc.weight, lr=1e-4)
        self.count += 1
        if args.Ntime:
            if self.count == args.backstep:
                self.count = 0
                if self.dim_hook is not None and labels is not None:
                    # 隐层输出梯度
                    LB = labels.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape)
                    #grad_output = (self.sumspike).mul(LB)/10 #STP
                    #grad_output = (1-self.sumspike/20).mul(LB) #STD
                    grad_output = LB
                    # grad_output = torch.zeros(LB.shape).cuda()
                    #grad_output = self.spike.mul(LB) #None
                    # grad_output.zero_() # gradient set 0, donot modify the weights
                    # 更新局部梯度
                    optimizer.zero_grad()
                    # print(grad_output.max())
                    # print((grad_output/self.spike_window*args.backstep).max())
                    self.spike.backward(gradient=grad_output/self.spike_window*args.backstep, retain_graph=True)
                    # print(1)
                    optimizer.step()
                    # print(self.fc.weight.max())
        else:
            if self.n<args.n and self.count >= args.backstep:
                self.n+=1
                if self.dim_hook is not None and labels is not None:
                    # print(self.count, self.n)
                    # 隐层输出梯度
                    LB = labels.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape)
                    #grad_output = (self.sumspike).mul(LB)/10 #STP
                    #grad_output = (1-self.sumspike/20).mul(LB) #STD
                    grad_output = LB
                    # grad_output = torch.zeros(LB.shape).cuda()
                    #grad_output = self.spike.mul(LB) #None
                    # grad_output.zero_() # gradient set 0, donot modify the weights
                    # 更新局部梯度
                    optimizer.zero_grad()
                    self.spike.backward(gradient=grad_output/self.spike_window*args.n, retain_graph=True)
                    optimizer.step()

        if self.time_counter == self.spike_window:
            self.time_counter = 0
            self.count = 0
            self.n = 0
                # self.loacl_gradient_pytorch(self.spike, grad_output)

        #cooperation = self.spike.sum(1).float()/(self.spike.shape[1])
        #cooperation = cooperation.unsqueeze(1).expand(self.spike.shape[0],self.spike.shape[1])
        #alpha = 0.5
        #self.spike = self.spike * (cooperation * alpha +1)
        return self.spike

    def reset_weights(self):
        nn.init.kaiming_uniform_(self.B)
        self.B.requires_grad = False

    # def loacl_gradient_pytorch(self, output, grad_output):
    #     output.backward(gradient=grad_output, retain_graph=False)


class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook,label_features, train_mode, batch_size, spike_window):
        super(CNN_block, self).__init__()
        self.spike_window = spike_window
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        # self.conv_Norm = nn.Sequential(
        #             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                       stride=stride, padding=padding, bias=bias),
        #             nn.BatchNorm2d(out_channels)
        # )
        self.act = Activation(activation)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.dim_hook=dim_hook
        if dim_hook is not None:
            self.B = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def forward(self, x, labels, y):
        # if False:
        if self.time_counter == 0:
            self.mem = torch.zeros((int(self.batch_size/use_gpu_num), self.out_channels, x.size()[-2], x.size()[-1])).cuda()  # /4表示4个gpu同时工作
            self.spike = torch.zeros((int(self.batch_size/use_gpu_num), self.out_channels, x.size()[-2], x.size()[-1])).cuda()
            self.sumspike = torch.zeros((int(self.batch_size/use_gpu_num), self.out_channels, x.size()[-2], x.size()[-1])).cuda()
        
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.conv, x, self.mem, self.spike)
        # self.mem, self.spike = mem_update(self.conv_Norm, x, self.mem, self.spike)
        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if self.dim_hook is not None and labels is not None:
                # 隐层输出梯度
                grad_output = labels.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape) 
                # grad_output.zero_()  # gradient set 0, donot modify the weights
                # 更新局部梯度
                self.loacl_gradient_pytorch(self.spike, grad_output)
                # print(self.conv.weight[0,0,0,:10])
                # # self.conv.weight.data = self.conv.weight.data - 1 * self.conv.weight.grad
                # print(self.conv.weight.grad.sum())
                # print(self.conv.weight[0,0,0,:10])
        output = self.pool(self.spike)
        return output
    
    def reset_weights(self):
        nn.init.kaiming_uniform_(self.B)
        self.B.requires_grad = False
    
    def loacl_gradient_pytorch(self, output, grad_output):
        output.backward(gradient=grad_output, retain_graph=False)


class C_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook,label_features, train_mode, batch_size, spike_window):
        super(C_block, self).__init__()
        self.spike_window = spike_window
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding, bias=bias)
        self.act = Activation(activation)
        self.pool = nn.AvgPool1d(kernel_size=kernel_size)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.dim_hook=dim_hook
        if dim_hook is not None:
            self.B = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def forward(self, x, labels, y):  # y 可以去掉
        if self.time_counter == 0:
                self.mem = torch.zeros((self.batch_size, self.out_channels, x.size()[-1])).cuda()
                self.spike = torch.zeros((self.batch_size, self.out_channels, x.size()[-1])).cuda()
                self.sumspike = torch.zeros((self.batch_size, self.out_channels, x.size()[-1])).cuda()
       
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.conv, x, self.mem, self.spike)

        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if self.dim_hook is not None and labels is not None:
                # 隐层输出梯度
                grad_output = labels.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape) 
                # 更新局部梯度
                self.loacl_gradient_pytorch(self.spike, grad_output)
                # print(self.conv.weight[0,0,:10])
                # # self.conv.weight.data = self.conv.weight.data - 1 * self.conv.weight.grad
                # print(self.conv.weight.grad[0,0,:10])
                # print(self.conv.weight[0,0,:10])

        output = self.pool(self.spike)
        return output

    def reset_weights(self):
        nn.init.kaiming_uniform_(self.B)
        self.B.requires_grad = False
    
    def loacl_gradient_pytorch(self, output, grad_output):
        output.backward(gradient=grad_output, retain_graph=False)


class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError("=== ERROR: activation " + str(activation) + " not supported")

    def forward(self, x):
        if self.act == None:
            return x
        else:
            return self.act(x)
