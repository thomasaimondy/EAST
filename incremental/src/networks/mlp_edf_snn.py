import sys
import torch
import numpy as np
from numpy import prod
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
sys.path.append("..")
import utils
spike_args = {}
spike_args['thresh'] = 0.5
spike_args['lens'] = 0.5
spike_args['decay'] = 0.5
from scipy.linalg import orth
import torch.nn.functional as F
# from module import FA_wrapper, TrainingHook

class Net(torch.nn.Module):

    def __init__(self,args, inputsize,taskcla,nlab, nlayers=3,nhid=40,pdrop1=0,pdrop2=0,spike_windows=2):
        super(Net,self).__init__()

        self.spike_window = spike_windows
        self.args = args
        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.labsize = nlab
        self.layers = nn.ModuleList()
        self.nlayers=nlayers

        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        # out_channels, kernel_size, padding =32, 5, 2
        # s=utils.compute_conv_output_size(size, kernel_size, padding=padding)
        # self.ec1 = torch.nn.Embedding(len(self.taskcla),out_channels)
        # self.ec1.weight.requires_grad = False
        # self.c1 = SpikeConv(ncha, out_channels, nlab, kernel_size, s, padding)

        self.fcs = nn.ModuleList()
        self.efcs = nn.ModuleList()

        if utils.B_plasticity == "TP":
            self.out = torch.zeros(args.sbatch, 10).cuda()
            self.out.requires_grad = False
        else:
            self.out = None

        # hidden layer
        for i in range(nlayers):
            if i == 0:
                efc = torch.nn.Embedding(len(self.taskcla),nhid)
                if self.args.Mask_type == "B":
                    efc.weight.requires_grad = False
                    efc.weight = Bclass.reset_weights_B(self.args, efc.weight)
                else:
                    efc.weight.requires_grad = False
                fc = SpikeLinear(self.args, size * size * ncha, nhid, nlab,layer=i)
            else:
                efc = torch.nn.Embedding(len(self.taskcla),nhid)
                if self.args.Mask_type == "B":
                    efc.weight.requires_grad = False
                    efc.weight = Bclass.reset_weights_B(self.args, efc.weight)
                else:
                    efc.weight.requires_grad = False
                fc = SpikeLinear(self.args, nhid, nhid, nlab, layer=i)
            self.fcs.append(fc)
            self.efcs.append(efc)

        # grad_LB = y.mm(self.B.view(-1, prod(self.B.shape[1:]))).view(self.spike.shape)
        # output layer
        if not args.multi_output:
            self.last= SpikeLinear(self.args, nhid, nlab, nlab,layer=-1) # 单头
        else:
            self.last=torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(SpikeLinear(self.args, nhid,n,nlab,layer=-1)) # 多头

        self.gate=torch.nn.Sigmoid()

        return

    def forward(self,t,x,laby,e=-1):
        # gc1 = self.gate(self.ec1(t)) #conv
        masks = self.mask(t) #fc

        for step in range(self.spike_window):
            # input
            # h = self.c1(x, laby, gc1)
            # h = h.detach()
            h = x.reshape(x.size(0), -1)
             # hidden
            for li in range(len(self.fcs)):
                gfc = masks[li]
                h = self.fcs[li](h, laby, gfc, t, e, self.out)
                if li == 0 and utils.train_mode=='train' and utils.trace_name is not None:
                    utils.TraceOfHidden.append((t, h)) 
                if li == 0 and utils.train_mode=='test' and utils.T == 9 and utils.trace_name is not None:
                    utils.TraceOfHiddenTest.append((t, h))
                h = h.detach()
            # output
            if self.args.multi_output:
                self.last[t](h,laby, gfc,t,e, self.out)
            else:
                self.last(h,laby, gfc,t, e, self.out)

        # output encoding
        if self.args.multi_output:
            y = self.last[t].sumspike / self.spike_window
        else:
            y = self.last.sumspike / self.spike_window
        # save output in inference
        if utils.train_mode=='test' and utils.T == 9 and utils.trace_name is not None:
            utils.TraceOfOutput.append((t, y)) 

        hidden_out = h
        return y, masks[0], hidden_out  #gc1, conv_out

    def mask(self,t):
        masks = []
        for li in range(len(self.efcs)):
            # Using sigmoid as a gate
            gfc = self.gate(100*self.efcs[li](t))
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
    def __init__(self, args, in_features, out_features, nlab, layer=None):
        super(SpikeLinear, self).__init__()
        self.args = args
        if layer!= -1:
            self.fc=torch.nn.Linear(in_features,out_features,bias=False)
            self.B = torch.empty(nlab, out_features).cuda()
            self.B = Bclass.reset_weights_B(self.args, self.B)
            self.B.requires_grad = False
        else:
            self.fc = torch.nn.Linear(in_features, out_features, bias=True)
        self.bn=torch.nn.BatchNorm1d(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.nlab = nlab
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.layer = layer
        self.spike_window = args.spike_windows
        # self.hook = TrainingHook(label_features=out_features, dim_hook=dim_hook, train_mode=args.B_plasticity)

        # Inspired from [Shan Yu, NMI, 2020]
        # self.input_size = self.in_features
        # self.output_size =  self.out_features
        # self.alpha = alpha
        # self.numclass = nlab
        self.P = torch.eye(self.in_features).cuda()

        self.P_old = torch.eye(self.in_features).cuda()
        self.P_new = torch.eye(self.in_features).cuda()

        self.input_ = None
        self.label_ = None
        self.h = None

        # initialization with zero y
        self.y_old = 0

    def update_p_1(self, input_, LB, y):
            r = torch.mean(input_,0,True)
            k = torch.mm(self.P, torch.t(r))
            temP = torch.mm(k,torch.t(k)) / (0.00001 + torch.mm(r,k))
            temP = torch.sub(self.P, temP)
            # temP = F.normalize(temP, p=2, dim=0)
            Ps_mini = F.interpolate(temP.unsqueeze(0).unsqueeze(0),size=[self.out_features,self.out_features])
            grad_output = torch.mm(LB, Ps_mini.squeeze(0).squeeze(0))
            self.P = temP
            return grad_output

    def update_p_standard(self, input_, LB, y):
        grad_output = 0
        # if 4 in y.argmax(dim=1):
        #     print()
        # Same y, approximate same P, not update P
        if (self.y_old - y).abs().sum() == 0:
            r = torch.mean(input_,0,True)
            k = torch.mm(self.P_old, torch.t(r))
            temP = torch.mm(k,torch.t(k)) / (0.00001 + torch.mm(r,k))
            if self.args.P_normalization:
                temP = F.normalize(temP, p=2, dim=1)
            temP = torch.sub(self.P_old, temP)
            Ps_mini = F.interpolate(temP.unsqueeze(0).unsqueeze(0),size=[self.out_features,self.out_features])
            grad_output = torch.mm(LB, Ps_mini.squeeze(0).squeeze(0))
            if self.args.P_proportion == '0.9-0.1':
                self.P_new = self.P_new * 0.9  + temP * 0.1 # Historical P is important
            elif self.args.P_proportion == '0.5-0.5':
                self.P_new = self.P_new * 0.5  + temP * 0.5 # Historical P is not very important
            elif self.args.P_proportion == '0.1-0.9':
                self.P_new = self.P_new * 0.1  + temP * 0.9 # Historical P is nonesless
            elif self.args.P_proportion == '0-1':
                self.P_new = temP # No historical P
            else:
                self.P_new = temP # No historical P
        # update P
        else:
            # in case of running only one else
            if self.P_new.abs().sum() == 0:
                self.P_new = self.P_old
            r = torch.mean(input_,0,True)
            k = torch.mm(self.P_new, torch.t(r))
            temP = torch.mm(k,torch.t(k)) / (0.00001 + torch.mm(r,k))
            if self.args.P_normalization:
                temP = F.normalize(temP, p=2, dim=1)
            temP = torch.sub(self.P_new, temP)
            Ps_mini = F.interpolate(temP.unsqueeze(0).unsqueeze(0),size=[self.out_features,self.out_features])
            grad_output = torch.mm(LB, Ps_mini.squeeze(0).squeeze(0))
            self.P_old = temP
            torch.nn.init.constant_(self.P_new, 0)
            self.y_old = y

        # Normalization
        # temP = F.normalize(temP, p=2, dim=0)*0.0006
        return grad_output

    def update_p_normalization(self, input_, LB, y):
        grad_output = 0
        # if 4 in y.argmax(dim=1):
        #     print()
        # Same y, approximate same P, not update P

        with torch.no_grad():
            old_length = (LB.mean(0) * LB.mean(0)).sum()

        if (self.y_old - y).abs().sum() == 0:
            r = torch.mean(input_,0,True)
            k = torch.mm(self.P_old, torch.t(r))
            temP = torch.mm(k,torch.t(k)) / (0.00001 + torch.mm(r,k))
            if self.args.P_normalization:
                temP = F.normalize(temP, p=2, dim=1)
            temP = torch.sub(self.P_old, temP)
            Ps_mini = F.interpolate(temP.unsqueeze(0).unsqueeze(0),size=[self.out_features,self.out_features])
            grad_output = torch.mm(LB, Ps_mini.squeeze(0).squeeze(0))
            if self.args.P_proportion == '0.9-0.1':
                self.P_new = self.P_new * 0.9  + temP * 0.1 # Historical P is important
            elif self.args.P_proportion == '0.5-0.5':
                self.P_new = self.P_new * 0.5  + temP * 0.5 # Historical P is not very important
            elif self.args.P_proportion == '0.1-0.9':
                self.P_new = self.P_new * 0.1  + temP * 0.9 # Historical P is nonesless
            elif self.args.P_proportion == '0-1':
                self.P_new = temP # No historical P
            else:
                self.P_new = temP # No historical P
        # update P
        else:
            # in case of running only one else
            if self.P_new.abs().sum() == 0:
                self.P_new = self.P_old
            r = torch.mean(input_,0,True)
            k = torch.mm(self.P_new, torch.t(r))
            temP = torch.mm(k,torch.t(k)) / (0.00001 + torch.mm(r,k))
            if self.args.P_normalization:
                temP = F.normalize(temP, p=2, dim=1)
            temP = torch.sub(self.P_new, temP)
            Ps_mini = F.interpolate(temP.unsqueeze(0).unsqueeze(0),size=[self.out_features,self.out_features])
            grad_output = torch.mm(LB, Ps_mini.squeeze(0).squeeze(0))
            self.P_old = temP
            torch.nn.init.constant_(self.P_new, 0)
            self.y_old = y

        with torch.no_grad():
            new_length = (grad_output.mean(0) * grad_output.mean(0)).sum()

        # Normalization
        # temP = F.normalize(temP, p=2, dim=0)*0.0006

        if new_length == 0:
            result = grad_output
        else:
            result = grad_output / new_length * old_length

        return result

    # t: current task, x: input, y: output, e: epochs
    def forward(self,x,y,h_mask,t,e, out):
        self.input_ = x
        self.label_ = y
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_features)).cuda()
            self.spike = torch.zeros((batchsize, self.out_features)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_features)).cuda()
            self.block_weights = torch.empty(self.fc.weight.data.size()).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, self.bn, x, self.mem, self.spike)
        # A mask to spikes
        if self.args.mask and self.layer!=-1:
            self.spike = self.spike * h_mask.expand_as(self.spike)
        self.sumspike += self.spike

        # y=None for inference
        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if y is not None:
                # Hidden layers
                if self.layer != -1:
                    grad_LB = y.mm(self.B.view(-1, prod(self.B.shape[1:]))).view(self.spike.shape)
                    if self.args.B_plasticity == 'LTP':
                        # More spikes more plasticity
                        err = (self.sumspike).mul(grad_LB)
                    elif self.args.B_plasticity == 'LTD':
                        # More spikes less plasticity
                        err = (self.spike_window - self.sumspike).mul(grad_LB)
                    elif self.args.B_plasticity == 'LB_decay':
                        # Same plasticity more or less spikes
                        err = grad_LB * torch.exp(torch.Tensor([-e])).cuda()
                    elif self.args.B_plasticity == 'Err':
                        err = self.sumspike / self.spike_window - grad_LB
                    elif self.args.B_plasticity == 'LB':
                        err = grad_LB
                    else:
                        err = grad_LB
                    # change class or not
                    if utils.LBP_mode == 'Static_N':
                        grad_output = err
                    if utils.LBP_mode == 'Adaptive_N':
                        grad_output = self.update_p_standard(x, err, y) # standard p
                    if utils.LBP_mode == 'Adaptive_norm_N':
                        grad_output = self.update_p_normalization(x, err, y)  # normalized p
                    # grad_output = self.update_p_normalization(x, err, y) 

                    # print(self.P_new.mean())  ## thomas
                    if utils.B_plasticity == 'TP':
                        self.h = self.spike
                    else:
                        self.spike.backward(gradient = grad_output, retain_graph=False)
                # Output layers
                else:
                    # MSE
                    err = (self.sumspike / self.spike_window) - y
                    # CE
                    # err = (self.sumspike / self.spike_window).mul(y)
                    if utils.LBP_mode == 'Static_N':
                        grad_output = err
                    if utils.LBP_mode == 'Adaptive_N':
                        grad_output = self.update_p_standard(x, err, y) # standard p
                    if utils.LBP_mode == 'Adaptive_norm_N':
                        grad_output = self.update_p_normalization(x, err, y)  # normalized p
                    # grad_output = self.update_p(x, err, y)
                    # print(self.P_new.mean())  ## thomas
                    self.spike.backward(gradient = grad_output, retain_graph=False)
                    # sumspike_mean = self.sumspike / self.spike_window
                    # sumspike_mean.backward(gradient = grad_output, retain_graph=False)  # use sum_spike could be better !!

                if self.args.plot:
                    image = (self.sumspike / self.spike_window).squeeze()
                    plt.imshow(np.array(image.detach().cpu()))
                    plt.savefig('../res/image_h_'+str(self.layer)+'.png',dpi=400)

        return self.spike

class Bclass():
    def reset_weights_B(args,B):
        if args.B_type == 'Regions_Standard':
            torch.nn.init.constant_(B, 0)
            out,hid = B.shape
            w = int(hid/out)
            b = torch.empty(out, w).cuda()
            # nn.init.kaiming_uniform_(b)
            nn.init.uniform_(b, a=-0.5, b=0.5)
            for i in range(out):
                B[i,i*w:(i+1)*w] = b[i,:]
        elif args.B_type == 'Regions_Orthogonal_gain_10':
            torch.nn.init.constant_(B, 0)
            out,hid = B.shape
            w = int(hid/out)
            b = torch.empty(out, w).cuda()
            nn.init.orthogonal_(b, gain=10)
            for i in range(out):
                B[i,i*w:(i+1)*w] = b[i,:]
        elif args.B_type == 'Regions_Orthogonal_gain_1':
            torch.nn.init.constant_(B, 0)
            out,hid = B.shape
            w = int(hid/out)
            b = torch.empty(out, w).cuda()
            nn.init.orthogonal_(b, gain=1)
            for i in range(out):
                B[i,i*w:(i+1)*w] = b[i,:]
        elif args.B_type == 'Orthogonal':
            # torch.nn.init.constant_(self.B, 0)
            torch.nn.init.orthogonal_(B, gain=1)
        elif args.B_type == 'Uniform':
            nn.init.kaiming_uniform_(B)
        else:
            nn.init.kaiming_uniform_(B)
        # nn.init.uniform_(B, a=-0.1, b=0.1)   ### test
        # torch.nn.init.orthogonal_(B, gain=1)   ### test
        # nn.init.sparse_(B, sparsity=0.9)  ### test
        if args.plot:
            plt.imshow(np.array(B.detach().cpu()))
            plt.savefig('../res/image_B.png',dpi=400)
        return B


# TODO: This class is not finished
class SpikeConv(torch.nn.Module):
    def __init__(self, ncha, out_channels, nlab, kernel_size, s, padding):
        super(SpikeConv, self).__init__()
        self.conv=torch.nn.Conv2d(ncha, out_channels,kernel_size=kernel_size, padding=padding)
        self.bn=torch.nn.BatchNorm2d(ncha)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.B = torch.empty(nlab, out_channels, s, s).cuda()
        # nn.init.uniform_(self.B, a=-0.1, b=0.1)
        nn.init.orthogonal_(self.B)
        self.B.requires_grad = False
        self.s = s
        self.nlab = nlab
        self.out_channels = out_channels
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.batch_size = 64

    def forward(self,x,y,h_mask):
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_channels, self.s, self.s)).cuda()
            self.spike = torch.zeros((batchsize, self.out_channels, self.s, self.s)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_channels, self.s, self.s)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.conv, self.bn ,x, self.mem, self.spike)
        if self.args.mask:
            self.spike = self.spike*h_mask.view(1,-1,1,1).expand_as(self.spike)
        self.sumspike += self.spike

        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if y is not None:
                grad_output = y.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(self.spike.shape)
                self.spike.backward(gradient=grad_output, retain_graph=False)
        return self.spike

# Approximate BP
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

# Membrane potential
def mem_update(ops, batchnorm, x, mem, spike, lateral=None):
    # x = batchnorm(x)
    a = torch.sigmoid(ops(x))
    mem = mem * spike_args['decay'] * (1. - spike) + a
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike