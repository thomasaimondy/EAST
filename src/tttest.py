import sys
import torch
import numpy as np
from numpy import prod
import torch.nn as nn
import torch.optim as optim
import os
from OWMLayer import OWMLayer


class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()

        self.input  = torch.nn.Linear(2,2,bias=False)
        self.output = torch.nn.Linear(2,2,bias=False)
        self.B = torch.empty(2, 2).cuda()
        torch.nn.init.orthogonal_(self.B)
        # torch.nn.init.kaiming_uniform_(self.B)
        self.input.weight.data = torch.Tensor([[0.1,0.2],[0.3,0.4]]).t()
        self.output.weight.data = torch.Tensor([[0.5,0.6],[0.7,0.8]]).t()

    def forward(self, x, y):
        x1 = self.input(x)
        grad_output = y.mm(self.B.view(-1,prod(self.B.shape[1:]))).view(x1.shape)
        # print('LB:', grad_output)
        x1.backward(gradient=grad_output, retain_graph=False)
        x = self.output(x1.detach())
        return torch.nn.Sigmoid()(x), x1


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


    # train_mode = 'EDF_BP' # 
    train_mode ='EDF_BP(OWM)'
    # train_mode = 'None_BP(OWM)'
    print(train_mode)
    torch.manual_seed(0)


    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=1)
    mse = torch.nn.MSELoss()

    input_0 = torch.Tensor([[0.1, 0.3]]).cuda()
    label_0 = torch.Tensor([[1,0]]).cuda()
    input_1 = torch.Tensor([[0.7, 0.9]]).cuda()
    label_1 = torch.Tensor([[0,1]]).cuda()
    inputs= [input_0, input_1]
    labels = [label_0, label_1]

    orthogonal = OWMLayer([2, 2])
    model.train()
    print('train'+'-'*10)
    for i in range(2):
        print('task'+ str(i)+'--'*10)
        for j in range(10):
            y, hidden_out = model(inputs[i], labels[i])
            loss = mse(y, labels[i])

            # print('input:',inputs[i])
            # print('label:', labels[i])
            # print('output:',y)
            # print('loss:',loss)
            # print('delta_w1:', model.input.weight.grad.data) 
            loss.backward()

            if train_mode == 'EDF_BP(OWM)':
                grad_new = orthogonal.force_learn(model.output.weight, hidden_out, learning_rate=0.5)
                # print('delta_w2:', grad_new)
                model.output.weight.grad.data.zero_()  
            
            if train_mode == 'EDF_BP':
                # print('delta_w2:', model.output.weight.grad.data)
                print()
            
            if train_mode == 'None_BP(OWM)':
                model.input.weight.grad.data.zero_()
                grad_new = orthogonal.force_learn(model.output.weight, hidden_out, learning_rate=0.5)
                # print('delta_w2:', grad_new)
                model.output.weight.grad.data.zero_()
                
            optimizer.step()
            optimizer.zero_grad()
            # print('*'*10)

    model.eval()
    print('test'+'-'*10)    
    y, hidden_out = model(inputs[0], labels[0])
    
    similarity1 = torch.dist(y, labels[0], p=2) # 欧氏距离
    similarity2 = torch.dist(y, labels[0], p=1) # 汉密尔顿距离
    similarity3 = torch.cosine_similarity(y, labels[0]) # 余弦相似度
    print('欧氏距离：', similarity1)
    print('汉密尔顿距离：', similarity2)
    print('余弦相似度', similarity3)

#-----------------记录上一个任务的P
#  if train_mode == 'EDF_BP(OWM)':
#             # only for owm(BP)
#                 if i == 0:
#                     grad_new, P_ij = orthogonal.force_learn(model.output.weight, hidden_out, learning_rate=0.5)
#                     # print('delta_w2:', P_ij)
#                     # print('delta_w2:',model.output.weight.grad.data)
#                     # model.output.weight.grad.data.zero_()
#                     print('delta_w2:', model.output.weight.grad.data)
#                     # print()
#                 else:
#                     grad_new, P_ij = orthogonal.force_learn(model.output.weight, hidden_out, learning_rate=0.1)
#                     model.output.weight.grad.data =  torch.mm(P_pre[i-1], model.output.weight.grad.data)
#                     model.output.weight.data -= 0.5 * model.output.weight.grad.data
#                     print('delta_w2:', model.output.weight.grad.data)
#                     model.output.weight.grad.data.zero_()
#                 P_pre[i]=P_ij