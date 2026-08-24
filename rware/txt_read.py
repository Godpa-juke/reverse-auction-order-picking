import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(threshold=1500,linewidth=np.inf)
import time


filename  = ['ACT_8.txt','OP_8.txt','ORG_8.txt','PSO_8.txt','ACT_24.txt','OP_24.txt','ORG_24.txt','PSO_24.txt']
data_path = 'Sequencing_Result'
all_data  = list()

for file_idx in range(len(filename)):
    file_path = data_path + '/' + filename[file_idx]
    f = open(file_path,'r')
    lines = f.readlines()

    wanted_data = list()
    for line_idx in range(len(lines)):
        line = lines[line_idx].split(',')
        wanted_data.append([int(line[0]), int(line[1]), float(line[4])])
    all_data.append(wanted_data)
    f.close()

short_data = all_data[:(len(filename)//2)]
long_data  = all_data[(len(filename)//2):]


# X = list()
# Y = list()
#
# for data_idx in range(len(short_data)):
#     x_temp = list()
#     y_temp = list()
#
#     for idx in range(len(short_data[data_idx])):
#         x_temp.append(short_data[data_idx][idx][0])
#         y_temp.append(short_data[data_idx][idx][2])
#     X.append(x_temp)
#     Y.append(y_temp)
#
# fig = plt.figure(figsize=(10, 6))
# for i in range(len(short_data)):
#     plt.plot(X[i], Y[i],  alpha=0.5, linewidth=2)
#
# plt.title("Actual Time (8EA)")
# plt.legend(filename[:(len(filename)//2)])
# plt.xlabel("Batch")
# plt.ylabel("Time")
# plt.show()

X = list()
Y = list()

for data_idx in range(len(long_data)):
    x_temp = list()
    y_temp = list()

    for idx in range(len(long_data[data_idx])):
        x_temp.append(long_data[data_idx][idx][0])
        y_temp.append(long_data[data_idx][idx][1])
    X.append(x_temp)
    Y.append(y_temp)

fig = plt.figure(figsize=(10, 6))
for i in range(len(long_data)):
    plt.plot(X[i], Y[i],  alpha=0.5, linewidth=2)

plt.title("Operating Time (24EA)")
plt.legend(filename[(len(filename)//2):])
plt.xlabel("Batch")
plt.ylabel("Time")
plt.show()