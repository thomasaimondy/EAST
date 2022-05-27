import sys
import torch
sys.path.append("..")
import utils

class Net(torch.nn.Module):

    def __init__(self,args,inputsize,taskcla,nhid,nlayers=1):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.args = args
        self.nlayers = nlayers

        # out_channels, kernel_size, padding = 32, 5, 2
        # self.conv1 = torch.nn.Conv2d(ncha, out_channels, kernel_size=kernel_size, padding=padding)
        # s = utils.compute_conv_output_size(size, kernel_size, padding=padding)
        # s = s // 2
        # self.maxpool = torch.nn.MaxPool2d(2)

        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0)
        # self.fc1=torch.nn.Linear(ncha*s*s*out_channels,800)
        self.fc1 = torch.nn.Linear(ncha * size * size, nhid, bias=False)
        if self.nlayers > 1:
            self.fc2 = torch.nn.Linear(nhid,nhid, bias=False)
        if self.nlayers > 2:
            self.fc3=torch.nn.Linear(nhid,nhid, bias=False)
        
        self.last=torch.nn.ModuleList()
        for t, n in self.taskcla:
            if args.multi_output:
                self.last.append(torch.nn.Linear(nhid,n))
            else:
                self.last = torch.nn.Linear(nhid,n)
        return

    def forward(self,x):
        # h = self.maxpool(self.drop(self.relu(self.conv1(x))))
        h=x.view(x.size(0),-1)
        h=self.drop(self.relu(self.fc1(h)))

        # if utils.train_mode=='train' and utils.trace_name is not None:
            # utils.TraceOfHidden.append((utils.T, h)) 
        if utils.train_mode=='test' and utils.T == 9 and utils.trace_name is not None:
            utils.TraceOfHiddenTest.append((utils.u, h))

        if self.nlayers > 1:
            h=self.drop(self.relu(self.fc2(h)))
        if self.nlayers > 2:
            h=self.drop(self.relu(self.fc3(h)))
        y=[]
        for t, i in self.taskcla:
            if self.args.multi_output:
                y.append(self.last[t](h))
            else:
                y.append(self.last(h))

        if utils.train_mode=='test' and utils.T == 9 and utils.trace_name is not None:
            utils.TraceOfOutput.append((utils.u, self.last(h))) 

        return y
