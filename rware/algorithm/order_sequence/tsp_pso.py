# encoding:utf-8

'''
	Solution for Travelling Salesman Problem using PSO (Particle Swarm Optimization)
	Discrete PSO for TSP

	References: 
		http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.258.7026&rep=rep1&type=pdf
		http://www.cs.mun.ca/~tinayu/Teaching_files/cs4752/Lecture19_new.pdf
		http://www.swarmintelligence.org/tutorials.php

	References are in the folder "references" of the repository.
'''

from operator import attrgetter
import random, sys, time, copy
import numpy as np
import itertools

from rware.data.cost_maps import load_cost_map

# class that represents a graph
class Graph:

	def __init__(self, amount_vertices):
		self.edges = {} # dictionary of edges
		self.vertices = set() # set of vertices
		self.amount_vertices = amount_vertices # amount of vertices


	# adds a edge linking "src" in "dest" with a "cost"
	def addEdge(self, src, dest, cost = 0):
		# checks if the edge already exists
		if not self.existsEdge(src, dest):
			self.edges[(src, dest)] = cost
			self.vertices.add(src)
			self.vertices.add(dest)


	# checks if exists a edge linking "src" in "dest"
	def existsEdge(self, src, dest):
		return (True if (src, dest) in self.edges else False)


	# shows all the links of the graph
	def showGraph(self):
		print('Showing the graph:\n')
		for edge in self.edges:
			print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

	# returns total cost of the path
	def getCostPath(self, path):
		
		total_cost = 0
		for i in range(self.amount_vertices - 1):
			total_cost += self.edges[(path[i], path[i+1])]

		# add cost of the last edge
		total_cost += self.edges[(path[self.amount_vertices - 1], path[0])]
		return total_cost


	# gets random unique paths - returns a list of lists of paths
	def getRandomPaths(self, max_size):

		random_paths, list_vertices = [], list(self.vertices)

		initial_vertice = random.choice(list_vertices)
		if initial_vertice not in list_vertices:
			print('Error: initial vertice %d not exists!' % initial_vertice)
			sys.exit(1)

		list_vertices.remove(initial_vertice)
		list_vertices.insert(0, initial_vertice)

		for i in range(max_size):
			list_temp = list_vertices[1:]
			random.shuffle(list_temp)
			list_temp.insert(0, initial_vertice)

			if list_temp not in random_paths:
				random_paths.append(list_temp)

		return random_paths


# class that represents a complete graph
class CompleteGraph(Graph):

	# generates a complete graph
	def generates(self):
		for i in range(self.amount_vertices):
			for j in range(self.amount_vertices):
				if i != j:
					weight = random.randint(1, 10)
					self.addEdge(i, j, weight)


# class that represents a particle
class Particle:

	def __init__(self, solution, cost):

		# current solution
		self.solution = solution

		# best solution (fitness) it has achieved so far
		self.pbest = solution

		# set costs
		self.cost_current_solution = cost
		self.cost_pbest_solution = cost

		# velocity of a particle is a sequence of 4-tuple
		# (1, 2, 1, 'beta') means SO(1,2), prabability 1 and compares with "beta"
		self.velocity = []

	# set pbest
	def setPBest(self, new_pbest):
		self.pbest = new_pbest

	# returns the pbest
	def getPBest(self):
		return self.pbest

	# set the new velocity (sequence of swap operators)
	def setVelocity(self, new_velocity):
		self.velocity = new_velocity

	# returns the velocity (sequence of swap operators)
	def getVelocity(self):
		return self.velocity

	# set solution
	def setCurrentSolution(self, solution):
		self.solution = solution

	# gets solution
	def getCurrentSolution(self):
		return self.solution

	# set cost pbest solution
	def setCostPBest(self, cost):
		self.cost_pbest_solution = cost

	# gets cost pbest solution
	def getCostPBest(self):
		return self.cost_pbest_solution

	# set cost current solution
	def setCostCurrentSolution(self, cost):
		self.cost_current_solution = cost

	# gets cost current solution
	def getCostCurrentSolution(self):
		return self.cost_current_solution

	# removes all elements of the list velocity
	def clearVelocity(self):
		del self.velocity[:]


# PSO algorithm
class PSO:

	def __init__(self, graph, iterations, size_population, beta=1, alfa=1):
		self.graph = graph # the graph
		self.iterations = iterations # max of iterations
		self.size_population = size_population # size population
		self.particles = [] # list of particles
		self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
		self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))

		# initialized with a group of random particles (solutions)
		solutions = self.graph.getRandomPaths(self.size_population)

		# checks if exists any solution
		if not solutions:
			print('Initial population empty! Try run the algorithm again...')
			sys.exit(1)

		# creates the particles and initialization of swap sequences in all the particles
		for solution in solutions:
			# creates a new particle
			particle = Particle(solution=solution, cost=graph.getCostPath(solution))
			# add the particle
			self.particles.append(particle)

		# updates "size_population"
		self.size_population = len(self.particles)


	# set gbest (best particle of the population)
	def setGBest(self, new_gbest):
		self.gbest = new_gbest

	# returns gbest (best particle of the population)
	def getGBest(self):
		return self.gbest


	# shows the info of the particles
	def showsParticles(self):

		print('Showing particles...\n')
		for particle in self.particles:
			print('pbest: %s\t|\tcost pbest: %d\t|\tcurrent solution: %s\t|\tcost current solution: %d' \
				% (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
							particle.getCostCurrentSolution()))
		print('')


	def run(self):

		# for each time step (iteration)
		for t in range(self.iterations):

			# updates gbest (best particle of the population)
			self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

			# for each particle in the swarm
			for particle in self.particles:

				particle.clearVelocity() # cleans the speed of the particle
				temp_velocity = []
				solution_gbest = copy.copy(self.gbest.getPBest()) # gets solution of the gbest
				solution_pbest = particle.getPBest()[:] # copy of the pbest solution
				solution_particle = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle

				# generates all swap operators to calculate (pbest - x(t-1))
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_pbest[i]:
						# generates swap operator
						swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

						# append swap operator in the list of velocity
						temp_velocity.append(swap_operator)

						# makes the swap
						aux = solution_pbest[swap_operator[0]]
						solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
						solution_pbest[swap_operator[1]] = aux

				# generates all swap operators to calculate (gbest - x(t-1))
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_gbest[i]:
						# generates swap operator
						swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

						# append swap operator in the list of velocity
						temp_velocity.append(swap_operator)

						# makes the swap
						aux = solution_gbest[swap_operator[0]]
						solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
						solution_gbest[swap_operator[1]] = aux

				
				# updates velocity
				particle.setVelocity(temp_velocity)

				# generates new solution for particle
				for swap_operator in temp_velocity:
					if random.random() <= swap_operator[2]:
						# makes the swap
						aux = solution_particle[swap_operator[0]]
						solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
						solution_particle[swap_operator[1]] = aux
				
				# updates the current solution
				particle.setCurrentSolution(solution_particle)
				# gets cost of the current solution
				cost_current_solution = self.graph.getCostPath(solution_particle)
				# updates the cost of the current solution
				particle.setCostCurrentSolution(cost_current_solution)

				# checks if current solution is pbest solution
				if cost_current_solution < particle.getCostPBest():
					particle.setPBest(solution_particle)
					particle.setCostPBest(cost_current_solution)
		

if __name__ == "__main__":
	random.seed(42)
	num_rack = 1062
	data = load_cost_map()

	# creates the Graph instance
	input_list = list()
	optima_seq = list()
	sequencing_list = list()
	running_sequencing_list = list()

	for batch_idx in range(1000):
		# order_list_size = random.randint(5, 8)
		order_list_size = 8

		order_list = []
		cnt = 0
		while True:
			if len(order_list) >= order_list_size: break
			val = random.randint(1, num_rack)
			if val in order_list: continue
			else:
				order_list.append(val)
				cnt = cnt + 1
		input_list.append(order_list)

	for input_idx in range(len(input_list)):
		cur_list = sorted(input_list[input_idx])

		divied_list = [cur_list[0:8]]
		empty = []

		for divided in divied_list:
			result_seq = []
			main_length = 99999999999999999
			for seq in itertools.permutations(divided, len(divided)):
				sum_length = 0
				for idx in range(len(seq) - 1):
					sum_length += data[seq[idx]][seq[idx + 1]]

				if main_length >= sum_length:
					main_length = sum_length
					result_seq = seq
			empty = empty + list(result_seq)
		print(input_idx,empty)
		optima_seq.append(list(empty))

	# for input_idx in range(len(input_list)):
	# 	print(optima_seq[input_idx])

	for optima_idx in range(len(optima_seq)):
		cur_list = optima_seq[optima_idx]
		graph = Graph(amount_vertices=len(cur_list))
		for i in cur_list:
			for j in cur_list:
				if i == j:
					graph.addEdge(i, j, 0)
					graph.addEdge(j, i, 0)

				else:
					graph.addEdge(i, j, data[i][j])
					graph.addEdge(j, i, data[i][j])

		# creates a PSO instance
		iteration = 100
		population = 17
		beta = 0.2
		alfa = 0.5

		pso = PSO(graph, iterations=iteration, size_population=population, beta=beta, alfa=alfa)
		pso.run()  # runs the PSO algorithm
		pso.showsParticles()  # shows the particles
		running_sequencing_list.append(pso.getGBest().getPBest())


	# for input_idx in range(len(input_list)):
	# 	print(running_sequencing_list[input_idx])

	for idx in range(len(input_list)):
		cur_list = input_list[idx]
		graph = Graph(amount_vertices=len(cur_list))
		for i in cur_list:
			for j in cur_list:
				if i == j:
					graph.addEdge(i, j, 0)
					graph.addEdge(j, i, 0)

				else:
					graph.addEdge(i, j, data[i][j])
					graph.addEdge(j, i, data[i][j])

		# creates a PSO instance
		iteration  = 100
		population = 17
		beta = 0.2
		alfa = 0.5

		pso = PSO(graph, iterations=iteration, size_population=population, beta=beta, alfa=alfa)
		pso.run() # runs the PSO algorithm
		pso.showsParticles() # shows the particles
		sequencing_list.append(pso.getGBest().getPBest())

		# shows the global best particle
		print('gbest: %s | cost: %d\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))

	# for input_idx in range(len(input_list)):
	# 	print(sequencing_list[input_idx])

	all_sum_sequencing_list = 0
	all_sum_running_sequencing_list = 0
	all_sum_optima_seq = 0

	for idx in range(len(sequencing_list)):
		sum_sequencing_list = 0
		sum_running_sequencing_list = 0
		sum_optima_seq = 0

		cur_sequencing_list = 0
		cur_optima_seq = 0
		cur_running_sequencing_list = 0

		for seq_idx in range(len(input_list[idx])-1):
			cur_sequencing_list = data[sequencing_list[idx][seq_idx]][sequencing_list[idx][seq_idx + 1]]
			cur_optima_seq = data[optima_seq[idx][seq_idx]][optima_seq[idx][seq_idx+1]]
			cur_running_sequencing_list = data[running_sequencing_list[idx][seq_idx]][running_sequencing_list[idx][seq_idx+1]]

			sum_sequencing_list = sum_sequencing_list + data[sequencing_list[idx][seq_idx]][sequencing_list[idx][seq_idx + 1]]
			sum_optima_seq = sum_optima_seq + data[optima_seq[idx][seq_idx]][optima_seq[idx][seq_idx+1]]
			sum_running_sequencing_list = sum_running_sequencing_list + data[running_sequencing_list[idx][seq_idx]][running_sequencing_list[idx][seq_idx+1]]

		all_sum_sequencing_list = all_sum_sequencing_list + sum_sequencing_list
		all_sum_running_sequencing_list = all_sum_running_sequencing_list + sum_running_sequencing_list
		all_sum_optima_seq = all_sum_optima_seq + sum_optima_seq


		print(idx, cur_optima_seq, sum_optima_seq, all_sum_optima_seq, optima_seq[idx])
		print(idx, cur_running_sequencing_list, sum_running_sequencing_list, all_sum_running_sequencing_list,running_sequencing_list[idx])
		print(idx, cur_sequencing_list, sum_sequencing_list, all_sum_sequencing_list, sequencing_list[idx])
		print()
