################################################################################
#### E-flow for SNN continual learning
#### Only containing forward information propagation
#### 2021-06-10, Tielin Zhang, Hongxing Liu, Shuncheng Jia
################################################################################

import sys,time
import numpy as np
import torch
from tqdm import tqdm
import utils

torch.autograd.set_detect_anomaly(True)

class Appr(object):
    def __init__(self, model, nlab,nepochs=100,sbatch=16,lr=0.01,lr_min=5e-4,lr_factor=1,lr_patience=5,clipgrad=10000,lamb=0.75,smax=400, args=None):
        self.model=model
        self.args = args
        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad
        self.nlab=nlab

        self.ce=torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.BCE = torch.nn.BCELoss()
        self.optimizer=self._get_optimizer()

        self.mask_pre=None
        self.mask_back=None

        return

    # SGD
    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr) #original
        # return torch.optim.Adam(self.model.parameters(),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        ytrain = torch.zeros(ytrain.shape[0], self.nlab).cuda().scatter_(1, ytrain.unsqueeze(1).long(), 1.0).cuda()
        yvalid = torch.zeros(yvalid.shape[0], self.nlab).cuda().scatter_(1, yvalid.unsqueeze(1).long(), 1.0).cuda()
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            # hidden_out, masks = 
            self.train_epoch(t,xtrain,ytrain,e)
            clock1=time.time()
            # train_loss,train_acc=self.eval(t,xtrain,ytrain)
            train_loss, train_acc= 0, 0
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            if True:
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        lr = max(lr, self.lr_min)
                        print(' lr={:.1e}'.format(lr),end='')
                        # if lr<self.lr_min:
                            # print()
                            # break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)
                print()
                
                ## thomas ,quick training
                if valid_acc>0.99:
                    break
                # if valid_loss < 0.01:
                #     break


        # Restore best validation model
        utils.set_model_(self.model,best_model)

        if self.args.mask:
            # Activations mask
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            mask=self.model.mask(task)
            for i in range(len(mask)):
                mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
            if t==0:
                self.mask_pre=mask
            else:
                for i in range(len(self.mask_pre)):
                    self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

            # Weights mask
            self.mask_back={}
            for n,p in self.model.named_parameters():
                vals=self.model.get_view_for(n,self.mask_pre)      # 使用mask影响梯度更新 
                if vals is not None:
                    self.mask_back[n]=1-vals
                # if n == 'fcs.0.fc.weight':
                #     np.save('fcs.0.fc.weight',p.data)

        # return hidden_out, masks

    def train_epoch(self,t,x,y,e,thres_cosh=50,thres_emb=6):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Constrain embeddings
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
                p.data=torch.clamp(p.data,-thres_emb,thres_emb)   # 限制嵌入层的数值的变化在-6~6
        # Loop batches
        for i in tqdm(range(0,len(r),self.sbatch)):
            if i+self.sbatch<=len(r): 
                b=r[i:i+self.sbatch]
            else: 
                break
                # b=r[i:]
            # if x[b].size(0)==32:
            #     np.savetxt('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out/images_'+ str(t) + '.txt', np.array(x[b].view(x[b].size(0),-1).cpu()))
            #     print('Image are saved')
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            # Forward
            output, masks, hidden_out = self.model.forward(task,images,targets,e)

            loss=self.criterion(output,targets,masks)
            
            # Backward (SGD only on last layer)
            
            # loss.backward(retain_graph=True) #original
            
            if self.args.mask:
                # Restrict layer gradients in backprop     
                if t>0:
                    for n,p in self.model.named_parameters():
                        if n in self.mask_back:
                            p.grad.data*=self.mask_back[n]
            # This is not necessary, but keep it for the furture discussion
            # Compensate embedding gradients
            # for n,p in self.model.named_parameters():
            #     if n.startswith('e'):
            #         num=torch.cosh(torch.clamp(p.data,-thres_cosh,thres_cosh))+1
            #         den=torch.cosh(p.data)+1
            #         p.grad.data*=num/den

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)  # 防止梯度爆炸、消失
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            # torch.save()
        # return hidden_out, masks

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r):
                b=r[i:i+self.sbatch]
            else: 
                # b=r[i:]
                break
            with torch.no_grad():
                images=torch.autograd.Variable(x[b])
                targets=torch.autograd.Variable(y[b])
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda())

            # Forward
            output,masks,_=self.model.forward(task,images,None)
            loss=self.criterion(output,targets,masks)
            _,pred=output.max(1)
            targets=targets.max(1)[1]
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.item()*len(b)
            total_acc += hits.sum().data.item()
            # total_acc.append(hits.sum().data.cpu().numpy().item()/len(b))
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num
        # return total_loss / total_num, total_acc.mean()
        # return total_loss / total_num, total_acc.max()

    # pure CE without regularization (may add something latter)
    def criterion(self,outputs,targets,masks):
       
        # targets = targets.argmax(dim=1)
        # return self.ce(outputs,targets.long())

        return self.mse(outputs,targets)
        
########################################################################################################################
