import numpy as np
import matplotlib.pyplot as plt
import torch

# x = [[1,1],[2,2],[3,3]]
# y = [[0.9981, 0.2777], [0.9981, 0.3205], [0.9227, 0.6931]]
# labels = ['SGD','EWC','EDF-SNN']
# for i in range(len(x)):

#     plt.plot(x[i],y[i],'--',label=labels[i])
#     plt.scatter(x[i][0],y[i][0], color='r' )
#     plt.scatter(x[i][1],y[i][1], color='b')
    
# plt.scatter(x[0][0],y[0][0], color='r', label='Previous accuracy')
# plt.scatter(x[0][1],y[0][1], color='b', label='Current accuracy')
# plt.legend()
# plt.title('Changes in task accuracy of continuous learning')

# plt.savefig('single_out.png', dpi=400)

# w = torch.empty(10, 5)
# w = torch.randn(10, 5)
w = torch.Tensor(4, 5)
# w = torch.nn.init.sparse(w, sparsity=0.5)
# w = torch.where(w > 0, torch.tensor(0.1), torch.tensor(0.0))
# w = torch.nn.init.kaiming_uniform_(w)
# w = torch.nn.init.uniform_(w, a=-0.1,b=0.1)
# print(w.mean(dim=1))
w = torch.nn.init.orthogonal_(w)
# w = w.reshape(4,-1)
print(w)
print(w.mm((w.t())))
