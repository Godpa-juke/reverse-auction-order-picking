import random
import numpy as np
from rware.algorithm.order_sequence.aco import *
# from rware.algorithm.order_sequence.tsp_pso import *

def aco_based_order_sequence(batch, cost_matrix):
    cur_cost_matrix = []
    cur_list = []
    for idx in range(len(batch)):
        cur_list.append(batch[idx][0])

    for i in cur_list:
        temp_list = []
        for j in cur_list:
            temp_list.append(cost_matrix[i][j])
        cur_cost_matrix.append(temp_list)

    rank = len(cur_list)
    aco = ACO(20, 6, 1.0, 10.0, 0.5, 10, 2)

    graph = Graph(cur_cost_matrix, rank)
    path, cost = aco.solve(graph)

    new_path = [batch[path[path_idx]] for path_idx in range(len(path))]
    print('original seq : ',batch)
    print('modified seq : ',new_path)

    return new_path, cost

# def pso_based_order_sequence(batch, cost_matrix):
#     cur_list = []
#     for idx in range(len(batch)):
#         cur_list.append(batch[idx][0])
#
#     print(cur_list)
#
#     graph = Graph(amount_vertices=len(cur_list))
#     for i in cur_list:
#         for j in cur_list:
#             if i == j:
#                 graph.addEdge(i, j, 0)
#                 graph.addEdge(j, i, 0)
#
#             else:
#                 graph.addEdge(i, j, cost_matrix[i][j])
#                 graph.addEdge(j, i, cost_matrix[i][j])
#
#     # creates a PSO instance
#     iteration = 100
#     population = 17
#     beta = 0.2
#     alfa = 0.5
#
#     pso = PSO(graph, iterations=iteration, size_population=population, beta=beta, alfa=alfa)
#     pso.run()  # runs the PSO algorithm
#     pso.showsParticles()  # shows the particles
#     sequencing_list.append(pso.getGBest().getPBest())
#
#     return pso.getGBest().getPBest(), pso.getGBest().getCostPBest()

