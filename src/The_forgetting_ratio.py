import numpy as np
file1 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-0.txt')
file2 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-1.txt')
file3 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-2.txt')
file4 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-3.txt')
file5 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-4.txt')
file6 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-5.txt')
file7 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-6.txt')

file8 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-10.txt')
file9 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-13.txt')
file10 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-25.txt')
file11 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-0.txt')
file12 = open('/home/user/liuhongxing/hat-increment_test-modify/res/pmnist_random_5-0.txt')

files = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]

task_acc = []  


for n, file in enumerate(files):
    data = file.readlines() #读取文档数据
    print(file.name)
    for j, num in enumerate(data):
        sum = 0
        for i in num.split(' '):
            sum += float(i)
        task_acc.append(sum/(j+1))
task_acc = np.array(task_acc).reshape(10,10)
print(task_acc.mean(axis=0))

# 十次的平均值 [0.10117    0.10026    0.10013667 0.1004225  0.100476   0.10082167  0.10017571 0.10054125 0.10066111 0.100581  ]