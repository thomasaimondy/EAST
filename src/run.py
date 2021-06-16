##########################################################################################
#### Code for Incremental learning of SNNs using E-flow
#### Multiple benchmark datasets and ANNs using SOTA algorithms (EWC, SGD, Joint, ...)
#### 2021-06-09， CASIA, Tielin Zhang, Shuncheng Jia, Hongxing Liu
#### Based on architectures of OWM [Shan, Yu, NMI, 2020] and HAT [Serrà, 2018, ICML]
#########################################################################################

import sys,os,argparse,time
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)') #(0, 1,2,3,4,5,6,10,13,25)
parser.add_argument('--mini', action='store_true', default=True, help='the mini dataset')
parser.add_argument('--experiment',default='mnist10_10',type=str,required=False,choices=['mnist2','pmnist','cifar','mixture','hwdb','mnist10_10','mnist5_10'],help='(default=%(default)s)')

parser.add_argument('--approach',default='edfsnn',type=str,required=False,choices=['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                            'imm-mode','sgd-restart',
                                                                            'joint','hat','hat-test','edf',  'edfsnn', 'sgdsnn'],help='(default=%(default)s)') #expectation-assisted disperse flow
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=400,type=int,required=False,help='(default=%(default)d)') # enough iteration times is important
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_factor',default=1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--gpu',type=str,default='1',help='(default=%(default)s)')
parser.add_argument('--multi_output', action='store_true', default=False, help='the type of ouput layer')
parser.add_argument('--P_normalization', action='store_true', default=False, help='the normalization of P vector')
parser.add_argument('--P_proportion',type=str,default='0.9-0.1',required=False, choices=['0.9-0.1','0.5-0.5','0.1-0.9', '0-1'],help='(default=%(default)s)')
parser.add_argument('--spike_windows',type=int,default=20,help='(default=%(default)s)')
parser.add_argument('--mask', action='store_true', default=True)
parser.add_argument('--Mask_type',type=str,default='Embed',required=False, choices=['B','Embed'],help='(default=%(default)s)')
parser.add_argument('--B_type',type=str,default='Regions_Standard',required=False, choices=['Regions_Standard','Regions_Orthogonal_gain_10','Regions_Orthogonal_gain_1','Orthogonal','Uniform'],help='(default=%(default)s)')
parser.add_argument('--B_plasticity',type=str,default='LTP',required=False, choices=['LTP','LTD','LB','Err'],help='(default=%(default)s)')
parser.add_argument('--nhid',type=int,default=100,help='(default=%(default)d)')
parser.add_argument('--plot', action='store_true', default=False, help='plot inner states')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.output=='':
    if args.multi_output:
        args.output='../res/'+args.experiment+'_'+args.approach + '_' + '*******' + '_多头' +'.txt'
    else:
        args.output='../res/'+args.experiment+'_'+args.approach + '_' + 'OWM' + '_单头' +'.txt' #args.seed str(5)
else:
    if not os.path.isdir('../res/'+args.output):
        os.makedirs('../res/'+args.output)
    args.output='../res/'+args.output+'/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'.txt'
print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)
f = open(args.output.split('.txt')[0], 'w')
f.write('pid:' + str(os.getpid()) + '\n')
f.write(str(vars(args)).replace(',', '\n'))
f.close()

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='mnist10':
    from dataloaders import mnist10 as dataloader
elif args.experiment=='mnist5':
    from dataloaders import mnist5 as dataloader
elif args.experiment=='mnist5_10':
    from dataloaders import mnist5_10 as dataloader
elif args.experiment=='mnist10_10':
    from dataloaders import mnist10_10 as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader
elif args.experiment=='hwdb':
    from dataloaders import hwdb as dataloader

# Args -- Approach
if args.approach=='random':
    from approaches import random as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach=='sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach=='lwf':
    from approaches import lwf as approach
elif args.approach=='lfl':
    from approaches import lfl as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='imm-mean':
    from approaches import imm_mean as approach
elif args.approach=='imm-mode':
    from approaches import imm_mode as approach
elif args.approach=='progressive':
    from approaches import progressive as approach
elif args.approach=='pathnet':
    from approaches import pathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat as approach
elif args.approach=='joint':
    from approaches import joint as approach
elif args.approach=='edf':
    from approaches import edf as approach
elif args.approach=='edfsnn':
    from approaches import edfsnn as approach
# elif args.approach=='sgdsnn':
#     from approaches import edfsnn as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment=='hwdb' or args.experiment == 'mnist10_10' or args.experiment == 'mnist5' or args.experiment == 'mnist5_10':
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    elif args.approach=='edf':
        from networks import mlp_edf as network
    elif args.approach=='edfsnn':
        from networks import mlp_edf_snn as network
    elif args.approach == 'sgdsnn':
        from networks import sgd_snn as network
    else:
        from networks import mlp as network
else:
    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='edf':
        from networks import alexnet_edf as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
    else:
        from networks import alexnet as network

########################################################################################################################

# Load
print('Load data...')
data,taskcla,inputsize, labsize=dataloader.get(seed=args.seed, mini = args.mini)
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
if args.approach=='edfsnn' or args.approach=='sgdsnn':
    net=network.Net(args,inputsize,taskcla, labsize, nhid=args.nhid,spike_windows=args.spike_windows).cuda() #edfsnn
else:
    net=network.Net(args,inputsize,taskcla, nhid=args.nhid).cuda() #others
# utils.print_model_report(net)
# f = open(args.output.split('.txt')[0], 'a')
# f.write('\n\n model:\n' + str(vars(net)) + '\n')
# f.close()

if args.approach=='edfsnn' or args.approach=='sgdsnn':
    appr = approach.Appr(net, labsize ,nepochs=args.nepochs,lr=args.lr, lr_factor = args.lr_factor, args=args)
else:
    appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        print(min(ytrain),max(ytrain))
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    if args.plot:
        image = xtrain[0].squeeze()
        plt.imshow(np.array(image.cpu()))
        plt.savefig('../res/image_In.png',dpi=400)
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()
        if args.approach == 'edfsnn' or args.approach == 'sgdsnn':
            ytest = torch.zeros(ytest.shape[0], labsize).cuda().scatter_(1, ytest.unsqueeze(1).long(), 1.0).cuda()
        test_loss,test_acc=appr.eval(u,xtest,ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f')

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)

print('Done!')

for i in range(acc.shape[0]):
    print('{:5.1f}% '.format(100*acc[i,:i+1].mean()),end='')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
