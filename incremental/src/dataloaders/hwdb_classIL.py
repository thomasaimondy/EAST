import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
from tqdm import tqdm

class HWDB(Dataset):
    def __init__(self, txt_path, num_class=3755, transforms=None):
        super(HWDB, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            i = 0
            for line in f:
                if i > num_class:
                    break
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
                i = i + 1
                
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        # image = Image.open(self.images[index]).convert('RGB')
        image = self.images[index]
        label = self.labels[index]
        image = image[np.newaxis, :]
        # image = Image.fromarray(image)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)

def HWDB_classes_txt(root, out_path, num_class=None):
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


########################################################################################################################

def get(seed=0, mini = False):
    data = {}
    taskcla = []
    size = [1, 32, 32]
    if mini:
        labsize = 10
    else:
        labsize = 3755
    seeds = np.array(list(range(labsize)), dtype=int)

    if not os.path.isdir('../dat/binary_hwdbCIL/'):
        os.makedirs('../dat/binary_hwdbCIL')
        dat = {}
        # root = '/home/user/liuhongxing/BRP-SNN-origin/DATASETS/HWDB'
        root = '/home/user/jiashuncheng/E-Flow/dat/HWDB'
        dat['train'] = HWDB(root+'/train_mat.txt', num_class=labsize, transforms=None)
        dat['test'] = HWDB(root+'/test_mat.txt', num_class=labsize, transforms=None)
        for i, r in enumerate(seeds):
            data[i] = {}
            data[i]['name'] = 'hwdbCIL-{:d}'.format(i)
            data[i]['ncla'] = labsize
            print(i)
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}
                for image, target in tqdm(loader):
                    aux = image.view(-1).numpy()
                    image = torch.FloatTensor(aux).view(size)
                    # Separate different samples into different tasks
                    if i == target.numpy()[0]:
                        data[i][s]['x'].append(image)
                        data[i][s]['y'].append(target.numpy()[0])
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('../dat/binary_hwdbCIL/'), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../dat/binary_hwdbCIL/'), 'data' + str(r) + s + 'y.bin'))
    else:
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = labsize
            data[i]['name'] = 'hwdbCIL-{:d}'.format(i)
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_hwdbCIL'), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_hwdbCIL'), 'data' + str(r) + s + 'y.bin'))

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
# 使用def HWDB_classes_txt获取并保存数据存储路径
# root1 = '/home/user/liuhongxing/BRP-SNN-origin/DATASETS/HWDB'
# root2 = '/data1/liuhongxing/HDWB/CHW_mat3755'

# root2 = '/data1/liuhongxing/HDWB/CHW_mat3755'
# HWDB_classes_txt(root2 + '/train_each_1000', root1 + '/train_1000_mat.txt')
# HWDB_classes_txt(root2 + '/test_each_1000', root1 + '/test_1000_mat.txt')