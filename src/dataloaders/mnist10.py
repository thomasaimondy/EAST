import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms

########################################################################################################################

def get(seed=0,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[1,28,28]
    labelsize=10
    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.MNIST('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.MNIST('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    # data[0]={}
    # data[0]['name']='mnist-0-4'
    # data[0]['ncla']=5
    # data[1]={}
    # data[1]['name']='mnist-5-9'
    # data[1]['ncla']=5

    for i in range(10):
        data[i] = {}
        data[i]['name'] = 'smnist-{:d}'.format(i)
        data[i]['ncla'] = 10
        
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        data[0][s]={'x': [],'y': []}
        data[1][s]={'x': [],'y': []}
        data[2][s]={'x': [],'y': []}
        data[3][s]={'x': [],'y': []}
        data[4][s]={'x': [],'y': []}
        data[5][s]={'x': [],'y': []}
        data[6][s]={'x': [],'y': []}
        data[7][s]={'x': [],'y': []}
        data[8][s]={'x': [],'y': []}
        data[9][s]={'x': [],'y': []}
        for image,target in loader:
            label=target.numpy()[0]
            data[label][s]['x'].append(image)
            data[label][s]['y'].append(label)

    # "Unify" and save
    for n in range(10):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'].clone()
        data[t]['valid']['y']=data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size,labelsize

########################################################################################################################
