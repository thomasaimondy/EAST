import sys
import torch
import numpy as np
from numpy import prod
import torch.nn as nn
import torch.optim as optim
import os
sys.path.append("/home/user/liuhongxing/hat-increment_test-weight distribution/src/networks")
import utils

spike_args = {}
spike_args['thresh'] = 0.5 #0.5
spike_args['lens'] = 0.5
spike_args['decay'] = 0.2
# spike_args['spike_window'] = 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nlab, nlayers=0,nhid=2000,pdrop1=0,pdrop2=0,spike_windows=2):
        super(Net,self).__init__()

        spike_args['spike_window'] = spike_windows
        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.labsize = nlab
        self.layers = nn.ModuleList()
        self.nlayers=nlayers

        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        out_channels, kernel_size, padding =32, 5, 2
        s=utils.compute_conv_output_size(size, kernel_size, padding=padding)
        self.ec1 = torch.nn.Embedding(len(self.taskcla),out_channels)
        self.ec1.weight.requires_grad = True  #False
        self.c1 = SpikeConv(ncha, out_channels, nlab, kernel_size, s, padding)

        self.fcs = nn.ModuleList()
        self.efcs = nn.ModuleList()

        for i in range(nlayers+1):
            if i == 0:
                efc = torch.nn.Embedding(len(self.taskcla),nhid)
                efc.weight.requires_grad = False #True
                fc = SpikeLinear(out_channels*s*s,nhid,nlab)
            else:
                efc = torch.nn.Embedding(len(self.taskcla),nhid)
                efc.weight.requires_grad = False #True
                fc = SpikeLinear(nhid,nhid,nlab)
            self.fcs.append(fc)
            self.efcs.append(efc)  #efc是嵌入层

        # output layer
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(SpikeLinear(nhid,n,nlab,layer=-1))
    
        self.gate=torch.nn.Sigmoid()

        return


    def forward(self,t,x,laby):

        gc1 = self.gate(self.ec1(t))
        masks = self.mask(t) #fc
        
        for step in range(spike_args['spike_window']):
            # input
            h = self.c1(x, laby, gc1)  # 对卷积后的输入做mask 即*gc1
            conv_out = h.reshape(h.size(0), -1)
            # hidden
            for li in range(len(self.fcs)):
                gfc = masks[li]
                h = h.reshape(h.size(0), -1)
                h = self.fcs[li](h,laby, gfc)        
               
            self.last[t](h,laby, gfc)

        # output encoding
                
        y = self.last[t].sumspike / spike_args['spike_window']  
            
        return y, gc1, conv_out #masks, hidden_out

    def mask(self,t):
        masks = []
        for li in range(len(self.efcs)):
            gfc = self.gate(400*self.efcs[li](t))   # gate using sigmoid, can set a scale factor where   400 这里
            masks.append(gfc)
        return masks
    

    def get_view_for(self,n,masks):
        for i in range(len(self.layers)):
            if n == 'fc'+str(i)+'.weight':
                if i == 0:
                    fcweight = masks[i].data.view(-1,1).expand_as(self.fcs[i].weight)
                    return fcweight
                else:
                    post=masks[i].data.view(-1,1).expand_as(self.fcs[i].weight)
                    pre=masks[i-1].data.view(1,-1).expand_as(self.fcs[i-1].weight)
                    return torch.min(post,pre)
            if n =='fc'+str(i)+'.bias':
                fcbias = masks[i].data.view(-1)
                return fcbias
        return None

class SpikeLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, nlab, layer=None):
        super(SpikeLinear, self).__init__()
        self.fc=torch.nn.Linear(in_features,out_features)
        self.bn=torch.nn.BatchNorm1d(in_features)
        self.B = torch.empty(nlab, out_features).cuda()
        # self.reset_weights_B()
        nn.init.uniform_(self.B, a=-0.1, b=0.1)  # 服从均匀分布
        self.B.requires_grad = False
        self.out_features = out_features
        self.nlab = nlab
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.layer = layer

    def forward(self,x,y,h_mask):
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_features)).cuda()
            self.spike = torch.zeros((batchsize, self.out_features)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_features)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, self.bn ,x, self.mem, self.spike)
        # if self.layer!=-1:
        #     self.spike = self.spike*h_mask.expand_as(self.spike)
        self.sumspike += self.spike

        if self.time_counter == spike_args['spike_window']:
            self.time_counter = 0             

        return self.spike

    def reset_weights_B(self):
        # nn.init.uniform_(self.B)
        # nn.init.kaiming_uniform_(self.B)
        # torch.nn.init.orthogonal_(self.B, gain=1)   ### test
        # nn.init.sparse_(self.B, sparsity=0.9)

        # torch.nn.init.eye_(self.B)

        torch.nn.init.constant_(self.B, 0)
        out,hid = self.B.shape
        w = int(hid/out)
        for i in range(out):
            nn.init.uniform_(self.B[i,i*w:(i+1)*w], a=-0.1, b=0.1)
            # nn.init.normal_(self.B[i,i*w:(i+1)*w], mean=0, std=0.1)
        self.B.requires_grad = False

class SpikeConv(torch.nn.Module):
    def __init__(self, ncha, out_channels, nlab, kernel_size, s, padding):
        super(SpikeConv, self).__init__()
        self.conv=torch.nn.Conv2d(ncha, out_channels,kernel_size=kernel_size, padding=padding)
        self.bn=torch.nn.BatchNorm2d(ncha)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.B = torch.empty(nlab, out_channels, s, s).cuda()
        # self.reset_weights_B()
        nn.init.uniform_(self.B, a=-0.1, b=0.1)
        self.B.requires_grad = False
        self.s = s
        self.nlab = nlab
        self.out_channels = out_channels
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0

    def forward(self,x,y,h_mask):
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_channels, self.s, self.s)).cuda()
            self.spike = torch.zeros((batchsize, self.out_channels, self.s, self.s)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_channels, self.s, self.s)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.conv, self.bn ,x, self.mem, self.spike)
        # self.spike = self.spike*h_mask.view(1,-1,1,1).expand_as(self.spike)
        self.sumspike += self.spike

        if self.time_counter == spike_args['spike_window']:
            self.time_counter = 0
            # if y is not None:
            #     # y = torch.zeros(y.shape[0], self.nlab).cuda().scatter_(1, y.unsqueeze(1).long(), 1.0).cuda()
            #     grad_output = y.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape)
            #     self.spike.backward(gradient=grad_output, retain_graph=False)  # use only local gradient
        #output = self.pool(self.spike) 
        return self.spike

    def reset_weights_B(self):
        # nn.init.uniform_(self.B)
        # nn.init.kaiming_uniform_(self.B)
        # torch.nn.init.orthogonal_(self.B, gain=1)   ### test
        # nn.init.sparse_(self.B, sparsity=0.9)

        # torch.nn.init.eye_(self.B)

        torch.nn.init.constant_(self.B, 0.001)
        out,hid = self.B.shape
        w = int(hid/out)
        for i in range(out):
            nn.init.uniform_(self.B[i,i*w:(i+1)*w], a=-0.1, b=0.1)
            # nn.init.normal_(self.B[i,i*w:(i+1)*w], mean=0, std=0.1)
        self.B.requires_grad = False

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

act_fun = ActFun.apply
def mem_update(ops, batchnorm, x, mem, spike, lateral=None, mask=None):

    if mask is not None:
        a = torch.mul(ops.weight,mask)
        a = torch.mm(x, a.t()) + ops.bias
        #a = batchnorm(a)
        a = torch.sigmoid(a)
    else:
        #x = batchnorm(x)
        a = torch.sigmoid(ops(x))  ##todo
        # a = ops(x)
    mem = mem * spike_args['decay'] * (1. - spike) + a
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike
