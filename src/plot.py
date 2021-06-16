import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import FuncFormatter
import xlrd
# % matplotlib inline


# 引入Excel库的xlrd

class Get_name():
    def __init__(self, sheet):  # sheet=0,1,...
        self.data = xlrd.open_workbook("E:\Study\Data\essaycode\hat-increment\src\lookup.xlsx")
        self.tables = []
        self.table = self.data.sheets()[sheet]
        for rown in range(self.table.nrows):
            array = {'exp_name': '', 'real_name': ''}
            array['exp_name'] = self.table.cell_value(rown, 0)
            array['real_name'] = self.table.cell_value(rown, 1)
            self.tables.append(array)

    def real_name(self, name):
        for i in range(len(self.tables)):
            if name.split('-')[0] == self.tables[i]['exp_name']: return self.tables[i]['real_name']


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                # L.append(os.path.join(root, file))
                if root not in L: L.append(root)
    return L

def file_name2(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
                # if root not in L: L.append(root)
    return L

def read_data(f):
    x = []
    for i in range(len(f)):
        x.append(float(f[9].split(' ')[i].split('\n')[0]))
    return x


def mydata(filename):
    file = file_name(filename)
    dicts = {}
    for i in file:
        # i = '../res/es_5_2'
        ii = file_name2(i)
        x_ = np.zeros((len(ii), 10))
        for k in range(len(ii)):
            f = open(ii[k], 'r')
            x_[k] = read_data(f.readlines())
        dicts[i.split('/')[-1]] = x_
    return dicts


def my_plot(dicts, drawlist, ylim=[0, 1], loc='upper right'):
    for i in range(len(dicts)):
        if list(dicts.keys())[i] in drawlist:
            a = dicts[list(dicts.keys())[i]].copy()
            a = a[0].tolist()
            a.reverse()
            l = list(dicts.keys())[i]
            plt.plot(a, label=name_.real_name(l))
    plt.legend(fontsize=14, loc=loc)
    plt.ylim(ylim)
    plt.xticks(np.arange(0, 10), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('# of permutation so far', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()

def my_plot_mean(dicts, drawlist, ylim=[0, 1], loc='upper right'):
    fig, ax = plt.subplots()
    for i in range(len(dicts)):
        if list(dicts.keys())[i] in drawlist:
            a = dicts[list(dicts.keys())[i]].copy()
            a_mean = a.mean(axis=0)
            a_std = a.std(axis=0)
            a_mean = a_mean[::-1]
            a_std = a_std[::-1]
            l = list(dicts.keys())[i]
            x = range(len(a_mean))
            ax.plot(x, a_mean, label=l)
            ax.fill_between(a_mean, (a_mean -  0.5* a_std), (a_mean + 0.5 * a_std),color=colors[i],alpha=.1)
    plt.legend(fontsize=14, loc=loc)
    plt.ylim(ylim)
    plt.xticks(np.arange(0, 10), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('# of permutation so far', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()

# name_ = Get_name(0)
colors = ['#1F77B4','#FF7F0E','b','r','g','#5E9BD7','#A4A4A6','#FCBF01', 'deeppink', 'olive']
dicts = mydata('../res')
# drawlist = ['es_10_2', 'es_5_2']
# my_plot(dicts, drawlist)
drawlist = ['joint_5', 'es_5_2', 'sgd_5']
my_plot_mean(dicts, drawlist, loc='lower right')
# name_ = Get_name(1)
# drawlist = ['es_1_2', 'es_2_2', 'es_3_2', 'es_4_2', 'es_5_2']
# my_plot(dicts, drawlist, loc='lower left')
# drawlist = ['es_5_1', 'es_5_2', 'es_5_3', 'es_5_4', 'es_5_5']
# my_plot(dicts, drawlist, loc='lower left')