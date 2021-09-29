# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------


------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import models
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import sys

# for i in sys.argv[:]:
#     if 'codename' in i : codename = i.split('=')[-1]
# filepath = 'output/' + codename.split('-')[0] + '/' + codename

# writer = SummaryWriter('logs')
loss_function = nn.MSELoss()

def train(args, device, train_loader, traintest_loader, test_loader):
    # torch.manual_seed(50)# 42 50  10
    

    for trial in range(1,args.trials+1):
        # Network topology
        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, spike_window=args.spike_window,  device=device, thresh=args.thresh, randKill=args.randKill, lens=args.lens, decay=args.decay)
        # model = nn.DataParallel(model)
        print(model)
        if args.cuda:
            model.cuda()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

        # Loss function
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,))


        filepath = 'model/' + args.codename.split('-')[0] + '/' + args.codename
        if os.path.exists(filepath+'/model.pth') and args.cont==True:
            checkpoint = torch.load(filepath+'/model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 1
            print('无保存模型，将从头开始训练！')

        for epoch in range(start_epoch, args.epochs + 1):
            # Training
            train_epoch(args, model, device, train_loader, optimizer, loss)

            # Compute accuracy on training and testing set
            print("\nSummary of epoch %d:" % (epoch))
            test_epoch(args, model, device, traintest_loader, loss, 'Train',epoch)
            test_epoch(args, model, device, test_loader, loss, 'Test',epoch)
            if args.cont=='True':
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, filepath+'/model.pth')


def train_epoch(args, model, device, train_loader, optimizer, loss):
    model.train()

    if args.freeze_conv_layers:
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False

    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        if batch_idx > 23:
            break
        data, label = data.to(device), label.to(device)#.unsqueeze(1)
        if args.regression:
            targets = label
        else:
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)
        optimizer.zero_grad()
        output = model(data, targets, args,optimizer=optimizer,batch_idx=batch_idx)
        # loss_val = loss_function(output, targets)
        loss_val = loss[0](output, loss[1](targets))
        readout(loss_val)
        optimizer.step()


def writefile(args, file):
    filepath = 'output/'+args.codename
    filetestloss = open(filepath + file, 'a')
    return filetestloss

def readout(loss_val):
    loss_val.backward()

def test_epoch(args, model, device, test_loader, loss, phase,epoch):
    model.eval()

    test_loss, correct = 0, 0
    # if args.dataset != 'tidigits':
    len_dataset = len(test_loader.dataset)
    # else:
    #     len_dataset = test_loader[1].shape[0]*test_loader[1].shape[1]
    counter = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            if args.regression:
                targets = label
            else:
                targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1,label.unsqueeze(1).long(), 1.0)

            output = model(data, None, args)

            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            if not args.regression:
                correct += pred.eq(label.view_as(pred).long()).sum().item()
            counter += 1
            if counter > 23:
                break

    loss = test_loss / (counter * args.batch_size)
    if not args.regression:
        acc = 100. * correct / (counter * args.batch_size)
        print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))


        filetestloss = writefile(args, '/testloss.txt')
        filetestacc = writefile(args, '/testacc.txt')
        filetrainloss = writefile(args, '/trainloss.txt')
        filetrainacc = writefile(args, '/trainacc.txt')

        if phase == 'Train':
            # writer.add_scalar('train_loss', loss, epoch)
            # writer.add_scalar('train_acc', acc, epoch)
            filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetrainacc.write(str(epoch) + ' ' + str(acc) + '\n')
        if phase == 'Test':
            # writer.add_scalar('test_loss', loss, epoch)
            # writer.add_scalar('test_acc', acc, epoch)
            filetestloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetestacc.write(str(epoch) + ' ' + str(acc) + '\n')
    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))
