import sys
import torch
import numpy as np
from numpy import prod
import torch.nn as nn
import torch.optim as optim
import os

spike_args = {}
spike_args['thresh'] = 0.5
spike_args['lens'] = 0.5
spike_args['decay'] = 0.2
spike_args['spike_window'] = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nlab, nlayers=1,nhid=2000,pdrop1=0.2,pdrop2=0.5):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.labsize = nlab
        self.layers = nn.ModuleList()
        self.nlayers=nlayers

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        self.fcs = nn.ModuleList()
        self.efcs = nn.ModuleList()
        for i in range(nlayers+1):
            if i == 0: # first layer
                efc = torch.nn.Embedding(len(self.taskcla),nhid)
                fc = SpikeLinear(ncha*size*size,nhid,nlab,efc=efc)
            else:
                # hidden layers
                efc = torch.nn.Embedding(len(self.taskcla),nhid)
                fc = SpikeLinear(nhid,nhid,nlab,efc=efc)
            self.fcs.append(fc)
            self.efcs.append(efc)
        # output layer
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(SpikeLinear(nhid,n,nlab))
        self.gate=torch.nn.Sigmoid()

        return

    def forward(self,t,x,laby):
        
        # masks = self.get_masks(t)
        masks = []
        for li in range(len(self.efcs)):
            gfc = self.gate(self.efcs[li](t))   # gate using sigmoid, or try multi a big 500
            masks.append(gfc)

        for step in range(spike_args['spike_window']):
            # input
            h=self.drop1(x.view(x.size(0),-1)) 
            # hidden
            for li in range(len(self.fcs)):
                h=self.drop2(self.relu(self.fcs[li](h,laby,t)))   # .detach() not work??
                with torch.no_grad():
                    gfc = masks[li]
                    a = gfc.expand_as(h)
                h=h*a
            # last
            self.last[t](h,laby,t)
        # last encoding
        # y=[]
        # for ti,i in self.taskcla:
        #     if ti<=t:
        #         h = self.last[ti].sumspike / spike_args['spike_window']
        #     else:
        #         h = 0
        #     y.append(h)
        y = self.last[t].sumspike / spike_args['spike_window']


        return y,masks

    def get_view_for(self,n,masks):
        for i in range(self.layers):
            if n == 'fc'+str(i)+'.weight':
                if i == 0:
                    fcweight = masks[i].data.view(-1,1).expand_as(self.fcs[i].weight)
                    return fcweight
                else:
                    post=mask[i].data.view(-1,1).expand_as(self.fcs[i].weight)
                    pre=mask[i-1].data.view(1,-1).expand_as(self.fcs[i-1].weight)
                    return torch.min(post,pre)
            if n =='fc'+str(i)+'.bias':
                fcbias = masks[i].data.view(-1)
                return fcbias
        return None
    
    def get_masks(self,t):
        masks = []
        for li in range(len(self.efcs)):
            gfc = self.gate(self.efcs[li](t))   # gate using sigmoid, or try multi a big 500
            masks.append(gfc)
        return masks


class SpikeLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, nlab, efc=None):
        super(SpikeLinear, self).__init__()
        self.fc=torch.nn.Linear(in_features,out_features)
        self.B = torch.empty(nlab, out_features).cuda()
        nn.init.uniform_(self.B, a=-0.1, b=0.1)
        self.B.requires_grad = False
        self.efc = efc
        self.out_features = out_features
        self.nlab = nlab
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0

    def forward(self,x,y,t):
        if self.time_counter == 0:
            batchsize = len(y)
            self.mem = torch.zeros((batchsize, self.out_features)).cuda()
            self.spike = torch.zeros((batchsize, self.out_features)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_features)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike)
        self.sumspike += self.spike

        if self.time_counter == spike_args['spike_window'] and self.efc is not None:
            self.time_counter = 0
            # use random matrix B
            # y = torch.zeros(y.shape[0], self.nlab).cuda().scatter(1, y.unsqueeze(1).long(), 1.0).cuda()
            # grad_output = y.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape) 
            # grad_output = grad_output*self.efc(t)
            # use embedding
            # self.spike.backward(gradient=grad_output, retain_graph=True)
        return self.spike

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
def mem_update(ops, x, mem, spike, lateral=None, mask=None):

    if mask is not None:
        a = torch.mul(ops.weight,mask)
        a = torch.mm(x, a.t()) + ops.bias
        a = torch.sigmoid(a)
    else:
        a = torch.sigmoid(ops(x))
    mem = mem * spike_args['decay'] * (1. - spike) + a # 看一下的输出是不是一直增大torch.sigmoid(ops(x))
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike