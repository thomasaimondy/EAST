# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

 "setup.py" - Setup configuration and dataset loading.

 Project: PM - Predictive Modification

 Authors:  liuhongxing

 增加了HWDB 数据集的读取包括：class HWDB(Dataset) 、def load_dataset_HWDB(args, kwargs) 和 def HWDB_classes_txt
 使用def HWDB_classes_txt获取并保存数据存储路径
------------------------------------------------------------------------------
"""


import torch
import torchvision
from torchvision import transforms,datasets
import numpy as np
import os
import sys
import random
import subprocess
#加载tidigits数据集
from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn.preprocessing import normalize
import os
import pickle
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import scipy.io as sio

class SynthDataset(torch.utils.data.Dataset):

    def __init__(self, select, type):
        self.dataset, self.input_size, self.input_channels, self.label_features = torch.load( './DATASETS/'+select+'/'+type+'.pt')

    def __len__(self):
        return len(self.dataset[1])

    def __getitem__(self, index):
        return self.dataset[0][index], self.dataset[1][index]

def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        #memory_load = get_gpu_memory_usage()
        #cuda_device = np.argmin(memory_load).item()
        #torch.cuda.set_device(cuda_device)
        #device = torch.cuda.current_device()
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    if args.dataset == "regression_synth":
        print("=== Loading the synthetic regression dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_regression_synth(args, kwargs)
    elif args.dataset == "classification_synth":
        print("=== Loading the synthetic classification dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_classification_synth(args, kwargs)
    elif args.dataset == "MNIST":
        print("=== Loading the MNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_mnist(args, kwargs)
    elif args.dataset == "CIFAR10":
        print("=== Loading the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10(args, kwargs)
    elif args.dataset == "CIFAR10aug":
        print("=== Loading and augmenting the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10_augmented(args, kwargs)
    elif args.dataset == "tidigits":
        print("=== Loading and augmenting the tidifits dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_tidigits(args, kwargs)
    elif args.dataset == "dvsgesture":
        print("=== Loading and augmenting the dvsgesture dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_gesture(args, kwargs)
    elif args.dataset == "HWDB":
        print("=== Loading and augmenting the HWDB dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_HWDB(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
    args.regression = (args.dataset == "regression_synth")

    return (device, train_loader, traintest_loader, test_loader)

def get_gpu_memory_usage():
    if sys.platform == "win32":
        curr_dir = os.getcwd()
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return gpu_memory

def load_dataset_regression_synth(args, kwargs):

    trainset = SynthDataset("regression","train")
    testset  = SynthDataset("regression", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features

    return (train_loader, traintest_loader, test_loader)

def load_dataset_classification_synth(args, kwargs):

    trainset = SynthDataset("classification","train")
    testset  = SynthDataset("classification", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features

    return (train_loader, traintest_loader, test_loader)

def load_dataset_mnist(args, kwargs):
    train_loader     = torch.utils.data.DataLoader(datasets.MNIST('/home/jiashuncheng/code/EDF-classification/DATASETS', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.batch_size,      shuffle=True ,drop_last = True, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/jiashuncheng/code/EDF-classification/DATASETS', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False, drop_last = True,**kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.MNIST('/home/jiashuncheng/code/EDF-classification/DATASETS', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False, drop_last = True,**kwargs)
    args.input_size     = 28
    args.input_channels = 1
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10(args, kwargs):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_cifar10 = transforms.Compose([transforms.ToTensor(),normalize,])

    root1 = '/mnt/lustre/xushuang4/jiashuncheng/code/newdrtp/DATASETS'
    train_loader     = torch.utils.data.DataLoader(datasets.CIFAR10(root1, train=True,  download=True, transform=transform_cifar10), batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root1, train=True,  download=True, transform=transform_cifar10), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.CIFAR10(root1, train=False, download=True, transform=transform_cifar10), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10_augmented(args, kwargs):
    #Source: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]]),])

    trainset = torchvision.datasets.CIFAR10('./DATASETS', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    traintestset = torchvision.datasets.CIFAR10('./DATASETS', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=args.test_batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR10('./DATASETS', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def read_data(path, n_bands, n_frames):
    overlap = 0.5

    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.waV') and file[0] != 'O':
                filelist.append(os.path.join(root, file))
    # filelist = filelist[:1002]

    n_samples = len(filelist)

    def keyfunc(x):
        s = x.split('/')
        return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
    filelist.sort(key=keyfunc)

    feats = np.empty((n_samples, 1, n_bands, n_frames))
    labels = np.empty((n_samples,), dtype=np.long)
    with tqdm(total=len(filelist)) as pbar:
        for i, file in enumerate(filelist):
            pbar.update(1)
            label = file.split('\\')[-1][0]  # if using windows, change / into \\
            if label == 'Z':
                labels[i] = np.long(0)
            else:
                labels[i] = np.long(label)
            rate, sig = wav.read(file)
            duration = sig.size / rate
            winlen = duration / (n_frames * (1 - overlap) + overlap)
            winstep = winlen * (1 - overlap)
            feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
            # feat = np.log(feat)
            final_feat = feat[:n_frames]
            final_feat = normalize(final_feat, norm='l1', axis=0)
            feats[i] = np.expand_dims(np.array(final_feat),axis=0)
        # feats[i] = feat[:n_frames].flatten() # feat may have 41 or 42 frames
        # feats[i] = feat.flatten() # feat may have 41 or 42 frames

    # feats = normalize(feats, norm='l2', axis=1)
    # normalization
    # feats = preprocessing.scale(feats)

    np.random.seed(42)
    p = np.random.permutation(n_samples)
    feats, labels = feats[p], labels[p]

    n_train_samples = int(n_samples * 0.7)
    print('n_train_samples:',n_train_samples)

    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    return train_set, train_set, test_set

def datatobatch(args,train_loader):
    temp, temp2 = [], []
    label, label2 = [], []
    for i, data in enumerate(train_loader[0]):
        if i % args.batch_size == 0 and i != 0:
            temp2.append(temp)
            label2.append(label)
            temp, label = [], []
            temp.append(data)
            label.append(train_loader[1][i])
        else:
            temp.append(data)
            label.append(train_loader[1][i])
    temp2 = torch.tensor(temp2)
    label2 = torch.tensor(label2)
    a = (temp2, label2)
    return a

class Tidigits(Dataset):
    def __init__(self,train_or_test,input_channel,n_bands,n_frames,transform=None, target_transform = None):
        super(Tidigits, self).__init__()
        self.n_bands = n_bands
        self.n_frames = n_frames
        dataname = '/home/jiashuncheng/code/EDF-classification/DATASETS/tidigits/packed_tidigits_nbands_'+str(n_bands)+'_nframes_' + str(n_frames)+'.pkl'
        if os.path.exists(dataname):
            with open(dataname,'rb') as fr:
                [train_set, val_set, test_set] = pickle.load(fr)
        else:
            print('Tidigits Dataset Has not been Processed, now do it.')
            train_set, val_set, test_set = read_data(path='/home/user/liuhongxing/BRP-SNN-origin/DATASETS/tidigits/isolated_digits_tidigits', n_bands=n_bands, n_frames=n_frames)#(2900, 1640) (2900,)
            with open(dataname,'wb') as fw:
                pickle.dump([train_set, val_set, test_set],fw)
        if train_or_test == 'train':
            self.x_values = train_set[0]
            self.y_values = train_set[1]

        elif train_or_test == 'test':
            self.x_values = test_set[0]
            self.y_values = test_set[1]
        elif train_or_test == 'valid':
            self.x_values = val_set[0]
            self.y_values = val_set[1]
        self.transform =transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label
    
    def __len__(self):
        return len(self.x_values)

class Gesture(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Gesture, self).__init__()
        mat_fname = '/mnt/lustre/xushuang4/jiashuncheng/code/newdrtp/DATASETS/DVS_gesture_100.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x_100']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x_100']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index, :, :]
        sample = torch.reshape(torch.tensor(sample), (sample.shape[0], 32, 32)).unsqueeze(0)
        label = self.y_values[index].astype(np.float32)
        label = torch.topk(torch.tensor(label), 1)[1].squeeze(0)
        return sample, label

    def __len__(self):
        return len(self.x_values)

class HWDB(Dataset):
    def __init__(self, txt_path, num_class=3755, transforms=None):
        super(HWDB, self).__init__()
        images = [] # 存储图片路径
        labels = [] # 存储类别名，在本例中是数字
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                samples = sio.loadmat(line)
                if 'data' in samples:
                    images.extend(samples['data'])
                    oo = samples['lables'].tolist()[0]
                elif 'train_data_each' in samples:
                    images.extend(samples['train_data_each'])
                    oo = samples['train_lable_each'].tolist()
                    oo = np.array(oo).flatten()
                elif 'test_data_each' in samples:
                    images.extend(samples['test_data_each'])
                    oo = samples['test_lable_each'].tolist()   
                    oo = np.array(oo).flatten()         
                labels.extend(oo)
        random.seed(10)
        random.shuffle(images)
        random.seed(10)
        random.shuffle(labels)
                # for i in samples['lables']:
                #     labels.append(i)
                #     labels.append(sample['lables'])
                # labels.append(int(line.split('/')[-1][6:-5]))  #当为mat文件时
        self.images = images
        self.labels = labels
        self.transforms = transforms # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        image = self.images[index]#Image.open(self.images[index]).convert('RGB') # 用PIL.Image读取图像
        label = self.labels[index]
        image = image[np.newaxis,:]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transforms is not None:
            image = self.transforms(image) # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)

def load_dataset_tidigits(args, kwargs):
    
    n_bands = 30
    n_frames = 30
    args.input_size = n_bands
    args.input_channels = 1
    args.label_features = 10

    train_dataset = Tidigits('train',args.input_channels, n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    traintest_dataset = Tidigits('valid',args.input_channels,n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    test_dataset = Tidigits('test',args.input_channels,n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle = True, drop_last = True)
    traintest_loader = torch.utils.data.DataLoader(dataset=traintest_dataset, batch_size=args.test_batch_size,shuffle = False,drop_last = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,shuffle = False,drop_last = True)

    

    return (train_loader, traintest_loader, test_loader)


def load_dataset_gesture(args, kwargs):

    args.input_size     = 32
    args.input_channels = 1
    args.label_features = 11

    train_dataset = Gesture('train', transform=transforms.ToTensor())
    test_dataset = Gesture('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = args.batch_size,shuffle = True,drop_last=True)
    traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = args.batch_size,shuffle = False,drop_last=True)

    return (train_loader, traintest_loader, test_loader)


def load_dataset_HWDB(args, kwargs):#
    
    # transform = transforms.Compose([transforms.ToTensor()
                                    # ]) # transforms.Normalize((0.0,),(1.0,))
    # transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])

    root = '/home/user/liuhongxing/BRP-SNN-origin/DATASETS/HWDB'
    # train_set = HWDB(root + '/train.txt', num_class=3755, transforms=transform)
    train_set = HWDB(root + '/train_mat.txt', num_class=3755, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    # train_set = HWDB(root + '/train_1000_mat.txt', num_class=3755, transforms=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True) # args.batch_size
    traintest_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # test_set = HWDB(root + '/test.txt', num_class=3755, transforms=transform)
    test_set = HWDB(root + '/test_mat.txt', num_class=3755, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    # test_set = HWDB(root + '/test_1000_mat.txt', num_class=3755, transforms=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    args.input_size     = 1024
    args.input_channels = 1
    args.label_features = 3755 #1000
  
    return (train_loader, traintest_loader, test_loader) # (test_loader,test_loader,test_loader)

def HWDB_classes_txt(root, out_path, num_class=None):  #可使用该函数获得mat的路径
    '''
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root) # 列出根目录下所有类别所在文件夹名
    if not num_class:		# 不指定类别数量就读取所有
        num_class = len(dirs)
    print(num_class)
    if not os.path.exists(out_path): # 输出文件路径不存在就新建
        f = open(out_path, 'w')
        f.close()
	# 如果文件中本来就有一部分内容，只需要补充剩余部分
	# 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                f.write(os.path.join(root, dir) + '\n')
                # files = os.listdir(os.path.join(root, dir))
                # for file in files:
                #     f.write(os.path.join(root, dir, file) + '\n')



# 使用def HWDB_classes_txt获取并保存数据存储路径
# root1 = '/home/user/liuhongxing/BRP-SNN-origin/DATASETS/HWDB'
# root2 = '/data1/liuhongxing/HDWB/CHW_mat3755'

# root2 = '/data1/liuhongxing/HDWB/CHW_mat3755'
# HWDB_classes_txt(root2 + '/train_each_1000', root1 + '/train_1000_mat.txt')
# HWDB_classes_txt(root2 + '/test_each_1000', root1 + '/test_1000_mat.txt')

