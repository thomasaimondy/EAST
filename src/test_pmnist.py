import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from dataloaders import mnist2 as dataloader
# a = torch.load(os.path.join(os.path.expanduser('../dat/binary_pmnist'), 'data' + str(1) + 'train' + 'y.bin'))
# b = torch.load(os.path.join(os.path.expanduser('../dat/binary_pmnist'), 'data' + str(9) + 'train' + 'y.bin'))

# print(a.shape, a[0:10])
# print(b.shape, b[0:10])

# for i in range(0,10):
    # weights_task0 = np.loadtxt('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out/hidden_out_'+str(i)+'_2.txt')
    # mask = np.loadtxt('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out/mask_'+str(i)+'_2.txt')
    # images = np.loadtxt('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out/images_'+ str(i)+'.txt')
    # print(weights_task0.shape)
    # for j in range(1):
    #     # weights_task = weights_task0[j].reshape((32,-1))[0] #(40,50)
    #     weights_task = weights_task0[j].reshape((40,50))
    #     # weights_task = np.where(weights_task>0.5, 1, 0)
    #     # weights_task = weights_task.reshape((28,28))
    #     print(weights_task.shape)
    #     # mask = mask.reshape((1,32)) # (40,50)
    #     mask = mask.reshape((40,50))
    #     # mask = np.where(mask>0.5, 1, 0)
    #     print(mask.shape)

    #     pure_out = weights_task/mask #[0,0]
    #     pure_out = np.where(pure_out>0.5,1,0)
    #     image = images[0].reshape((28,28))
    #     # image = np.where(image>0, 1, 0)
    #     # plt.imshow(image,cmap='gray')
    #     plt.xticks([])
    #     plt.yticks([])

    #     # plt.title('class '+ str(i))
    #     # plt.imshow(image,cmap='gray_r')#,
    #     # plt.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out1/image' + str(i)+ '.png',dpi=400)
    #     # plt.close()

        
    #     # plt.title('mask '+ str(i))
    #     # plt.imshow(mask,cmap='gray_r')#
    #     # plt.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out1/mask' + str(i)+ '.png',dpi=400)
    #     # plt.close()

    #     # plt.xticks([])
    #     # plt.yticks([])
    #     # plt.title('output_with_mask '+ str(i))
    #     # plt.imshow(weights_task,cmap='gray_r')#
    #     # plt.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out1/output_with_mask' + str(i)+ '.png',dpi=400)
    #     # plt.close()
        
    #     plt.title('pure_out '+ str(i))
    #     plt.imshow(pure_out,cmap='gray_r', aspect='auto')#
    #     plt.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out1/pure_out' + str(i)+ '.png', dpi=200)

    #     # estimator = PCA(n_components=10)
    #     # weights_task = estimator.fit_transform(weights_task)
    #     # weights_task = weights_task.T
    #     # print(weights_task.shape)
    #     # weights_task = estimator.fit_transform(weights_task)
    #     # print(weights_task.shape)

    #     # fig = sns.heatmap(image, xticklabels=False, yticklabels= False,cmap='Greys', cbar=True)
    #     # # fig = sns.heatmap(mask, xticklabels=False, yticklabels= False,cmap='Greys') # ,cbar=False
    #     # # fig = sns.heatmap(pure_out, xticklabels=False, yticklabels= False,cmap='Greys')

    #     # heatmap = fig.get_figure()

    #     # heatmap.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out1/conv_out_map_'+ str(i)+'_'+str(j)+'_2.png', dpi=400)
    #     # heatmap.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out1/mask_'+str(i)+'_2.png', dpi=400)
    #     # heatmap.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out1/conv_out_map_withoutmask_'+str(i)+'_'+str(j)+'_2.png', dpi=400)
    #     plt.close()


#画mnist原图
t = 0
data,taskcla,inputsize=dataloader.get(seed=0)
xtrain=data[t]['train']['x']
ytrain=data[t]['train']['y']
xvalid=data[t]['valid']['x']

clss = []
num = []
for n, c in enumerate(ytrain):
    if c==0:
        image = xtrain[n].view(28,28)
        # image = np.where(image>0.5,1,0)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(np.array(image),cmap='gray')
        plt.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/conv_out1/test_0.png',dpi=400)
        break

# # 画曲面图
# for i in range(0,3):
#     # weights_task = np.loadtxt('/home/user/liuhongxing/hat-increment_test-modify/src/fcs.0.fc.weight_'+str(3)+'.txt')
#     weights_task = np.loadtxt('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out/hidden_out_'+str(i)+'_2.txt')
#     print(weights_task.shape)  
    
#     fig = plt.figure()
#     # 创建3d图形的两种方式
#     # 将figure变为3d
#     ax = Axes3D(fig)

#     x = np.arange(0, 40, 1)
#     y = np.arange(0, 50, 1)

#     # 生成网格数据
#     X, Y = np.meshgrid(x, y)

#     # 计算每个点对的长度
#     R = np.sqrt(X ** 2 + Y ** 2)
#     # 计算Z轴的高度
#     Z = weights_task[0].reshape((50,40))

#     # 绘制3D曲面

#     # rstride:行之间的跨度  cstride:列之间的跨度
#     # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现

#     # cmap是颜色映射表
#     # from matplotlib import cm
#     # ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
#     # cmap = "rainbow" 亦可
#     # 我的理解的 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
#     # 你也可以修改 rainbow 为 coolwarm, 验证我的结论
#     ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
#     # 绘制从3D曲面到底部的投影,zdir 可选 'z'|'x'|'y'| 分别表示投影到z,x,y平面
#     # zdir = 'z', offset = -2 表示投影到z = -2上
#     # ax.contour(X, Y, Z, zdir = 'z', offset = -0.2, cmap = plt.get_cmap('coolwarm'))
#     # 设置z轴的维度，x,y类似
#     ax.set_zlim(-1, 1)
#     plt.savefig('/home/user/liuhongxing/hat-increment_test-modify/src/weight_visual/hidden_out/hidden_out_'+str(i)+'.png', dpi=400)
