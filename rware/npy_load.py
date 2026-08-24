from itertools import permutations, combinations
import time

import numpy as np

from rware.data.cost_maps import load_cost_map

np.set_printoptions(threshold=1500, linewidth=np.inf)
data = load_cost_map()

sample_list = [346, 365, 391, 403, 410, 450, 463, 486]

permutation_list = list(permutations(sample_list,len(sample_list)))

# result = 10000000000000
# for sample_idx in range(len(permutation_list)):
#     temp = 0
#     for element_idx in range(len(sample_list)-1):
#         temp = temp + data[permutation_list[sample_idx][element_idx]][permutation_list[sample_idx][element_idx+1]]
#
#     if temp < result:
#         result = temp
#         print(sample_idx, permutation_list[sample_idx], temp)
#         for element_idx in range(len(sample_list) - 1):
#             print(data[permutation_list[sample_idx][element_idx]][permutation_list[sample_idx][element_idx + 1]],end = ' ')
#         print()

order_batch = [[73, 1, 29], [64, 1, 27], [63, 1, 27], [52, 5, 21], [51, 3, 21], [124, 1, 35], [105, 1, 37], [82, 1, 27], [206, 1, 53], [204, 1, 51], [261, 1, 67], [163, 1, 43], [182, 1, 51], [191, 1, 53], [160, 1, 43], [110, 1, 37], [314, 1, 77], [319, 1, 75], [304, 1, 75], [346, 1, 85], [328, 1, 77], [267, 1, 69], [264, 1, 67], [276, 1, 69], [486, 1, 109], [463, 1, 107], [450, 1, 101], [489, 1, 109], [391, 1, 93], [403, 1, 91], [410, 1, 93], [365, 1, 85]]
p_list = [51, 52, 73, 82, 63, 64, 110, 105, 204, 206, 160, 163, 182, 191, 110, 124, 304, 261, 264, 267, 314, 319, 276, 328, 365, 410, 403, 346, 391, 450, 463, 486, 489]
result = []
for p_sample in p_list:
    for idx in range(len(order_batch)):
        if order_batch[idx][0] == p_sample:
            result.append(order_batch[idx])
            break

print(result)