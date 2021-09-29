import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image

class HWDB(Dataset):
    def __init__(self, txt_path, num_class=3755, incremental=False, transforms=None):
        super(HWDB, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，在本例中是数字
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            i = 0
            if incremental:
                num_class += 2

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
                oo = i * np.ones_like(oo)
                labels.extend(oo)
                i += 1
                if i == num_class:
                    break
        random.seed(10)
        random.shuffle(images)
        random.seed(10)
        random.shuffle(labels)

        ### test

        # random.seed(10)
        # random.shuffle(images)
        # random.seed(10)
        # random.shuffle(labels)

        # for i in samples['lables']:
        #     labels.append(i)
        #     labels.append(sample['lables'])
        # labels.append(int(line.split('/')[-1][6:-5]))  #当为mat文件时
        self.images = images
        self.labels = labels
        self.transforms = transforms  # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        image = self.images[index]  # Image.open(self.images[index]).convert('RGB') # 用PIL.Image读取图像
        label = self.labels[index]
        image = image[np.newaxis, :]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)

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

########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [1, 32, 32]
    labsize = 10

    nperm = 10
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir('../dat/binary_hwdb/'):
        os.makedirs('../dat/binary_hwdb')
        # Pre-load
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)
        dat = {}
        root = '/home/user/liuhongxing/BRP-SNN-origin/DATASETS/HWDB'
        dat['train'] = HWDB(root+'/train_mat.txt', num_class=10, incremental=False, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = HWDB(root+'/train_mat.txt', num_class=10, incremental=False, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'hwdb-{:d}'.format(i)
            data[i]['ncla'] = 10
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)
                    image = torch.FloatTensor(aux).view(size)
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('../dat/binary_hwdb/'), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../dat/binary_hwdb/'), 'data' + str(r) + s + 'y.bin'))
        print()

    else:

        # Load binary files
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 10
            data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_hwdb'), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_hwdb'), 'data' + str(r) + s + 'y.bin'))

    # Validation
    for t in data.keys():
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size, labsize

########################################################################################################################
