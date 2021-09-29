import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image

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
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)
    print(num_class)
    if not os.path.exists(out_path):
        f = open(out_path, 'w')
        f.close()
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

########################################################################################################################

def get(seed=0, mini = False):
    data = {}
    taskcla = []
    size = [1, 32, 32]
    if mini:
        labsize = 10
        nperm = int(10/2)
    else:
        labsize = 3754
        nperm = int(3754/2)
    seeds = np.array(list(range(labsize)), dtype=int)

    if not os.path.isdir('../dat/binary_hwdbTIL/'):
        os.makedirs('../dat/binary_hwdbTIL')
        dat = {}
        root = '/home/zhangtielin/maincode/DATASETS/HWDB'
        dat['train'] = HWDB(root+'/train_mat.txt', num_class=labsize, transforms=None)
        dat['test'] = HWDB(root+'/train_mat.txt', num_class=labsize, transforms=None)
        for i, r in enumerate(seeds):
            j = int(i/2)
            data[j] = {}
            data[j]['name'] = 'hwdbCIL-{:d}'.format(i)
            data[j]['ncla'] = nperm
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[j][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    image = torch.FloatTensor(aux).view(size)
                    # Separate different samples into different tasks, separate labels
                    if i == target.numpy()[0]:
                        data[j][s]['x'].insert(0,image)
                        data[j][s]['y'].insert(0,target.numpy()[0])
                    if i+1 == target.numpy()[0]:
                        data[j][s]['x'].append(0,image)
                        data[j][s]['y'].append(0,target.numpy()[0])
            for s in ['train', 'test']:
                data[j][s]['x'] = torch.stack(data[j][s]['x']).view(-1, size[0], size[1], size[2])
                data[j][s]['y'] = torch.LongTensor(np.array(data[j][s]['y'], dtype=int)).view(-1)
                torch.save(data[j][s]['x'],os.path.join(os.path.expanduser('../dat/binary_hwdbTIL/'), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[j][s]['y'],os.path.join(os.path.expanduser('../dat/binary_hwdbTIL/'), 'data' + str(r) + s + 'y.bin'))
            
    else:
        for i, r in enumerate(seeds):
            j = int(i/2)
            data[j] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[j]['ncla'] = nperm
            data[j]['name'] = 'hwdbCIL-{:d}'.format(i)
            for s in ['train', 'test']:
                data[j][s] = {'x': [], 'y': []}
                data[j][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_hwdbTIL'), 'data' + str(r) + s + 'x.bin'))
                data[j][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_hwdbTIL'), 'data' + str(r) + s + 'y.bin'))

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