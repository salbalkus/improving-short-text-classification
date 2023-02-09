from .candidate import Candidate
from random import shuffle, randint, random
from copy import deepcopy, copy
import pandas as pd
import numpy as np
from math import factorial
import openai


class Population:
	def __init__(self, example_space, solution_space, n_examples=3, size=8, max_tokens=1, fit_func="acc", engine="ada", mutation_rate=0.1, n_mutate_examples=3, max_mutate_length=64, mutate_stop_token="\n"):
		"""
		Constructs a new population of potential candidate solutions

		Args:
			example_space: list or pd.Series, containing all possible examples to consider as in-context examples
			n_examples: int, number of examples per candidates
			size: int, number of candidates to consider
			max_tokens: the expected number of tokens in the output
		"""
		if size % 2 > 0:
			raise Exception("population size must be even to engage in Tournament Selection")
		else:
			self.size = size
		self.example_space = example_space # Total genes available in the pool
		self.solution_space = solution_space # y-values associated with each example
		self.orig_example_space = example_space
		self.orig_solution_space = solution_space
		self.candidates = [] # List of candidates
		self.n_examples = n_examples # Number of examples per organism
		self.max_tokens = max_tokens
		self.best = None
		self.fit_func = fit_func
		self.engine = engine
		self.mutation_rate = mutation_rate
		self.max_mutate_length = max_mutate_length
		self.n_mutate_examples = n_mutate_examples
		self.mutate_stop_token = mutate_stop_token
		self.max_population_size = factorial(self.size - 1)

		# Randomize the example space and the solution space
		example_space = example_space.sample(n=example_space.size)
		solution_space = solution_space[example_space.index]
		example_space.reset_index(drop=True)
		solution_space.reset_index(drop=True)

		### Population Initialization ###
		# initialize a list of possible indices from which to sample
		k = 0
		samples = list(range(1, example_space.size))
		shuffle(samples)
		# Create the specified number of candidates to fill the population size
		while len(self.candidates) < self.size:
			# Here we construct a list of indices to sample from both the example and solution spaces in parallel
			select_samples = []
			# Sample the specified number of examples to the candidate
			while len(select_samples) < n_examples:
				select_samples.append(samples[k])
				k += 1
				# If we run out of samples, regenerate the possible indices to sample
				if k >= len(samples):
					# Reset the list of samples
					# Exclude indices being sampled in this round to avoid repeating the same allele in the same candidate
					samples = [x for x in list(range(1, example_space.size)) if x not in select_samples]
					shuffle(samples)
					k = 0
			self.candidates.append(Candidate(example_space[select_samples], solution_space[select_samples], output_tokens=self.max_tokens, fit_func=self.fit_func, engine=self.engine))

	def next_generation(self, X_test, y_test, new_test_set=False, immigrate=True):
		# Check and update any new population members fitness
		self.calc_fitness(X_test, y_test, new_test_set)

		# Add new members of the population likely to have improved fitness
		if not immigrate:
			self.tournament_selection()
			self.reproduce()
		else:
			self.tournament_selection4()
			self.reproduce_immigrate()

		# Evaluate new population members added
		self.calc_fitness(X_test, y_test)

		# Reassign the best candidate
		self.best = self.candidates[np.argmax(np.array([cand.fitness for cand in self.candidates]))]

	def calc_fitness(self, X_test, y_test, new_test_set=False):
		"""
		Update the fitness for each candidate.
		Args:
			X_test: pd.Series, the set of examples to evaluate
			y_test: pd.Series, labels of the X_test
			new_test_set: bool, whether to update the fitness of every candidate or just the unscored candidates. Use if test set has changed since last generation.
		"""
		for candidate in self.candidates:
			if candidate.fitness is None or new_test_set:
				print("Updating New Candidate Fitness")
				candidate.update_fitness(X_test, y_test)

	# TODO: Currently assumes higher fitness is better.
	# Modify to allow additional fitness metrics that may be minimized instead of maximized
	def tournament_selection(self):
		# Generate random pairing of candidates (evens face odds)
		samples = list(range(0, len(self.candidates)))
		shuffle(samples)
		# This implements a by for when population contains odd numbers
		if len(samples) % 2 != 0:
			samples.pop(-1)
		# For each random pair of candidates, remove less fit of the pair
		# Need to STORE the candidates to be kept since removing  affects
		# the length of the list
		keep_cands = []
		for k in range(0, len(samples), 2):
			if self.candidates[samples[k]].fitness < self.candidates[samples[k+1]].fitness:
				keep_cands.append(self.candidates[samples[k+1]])
			elif self.candidates[samples[k]].fitness > self.candidates[samples[k+1]].fitness:
				keep_cands.append(self.candidates[samples[k]])
			else:
				# If they are tied, choose randomly
				keep_cands.append(self.candidates[samples[randint(k, k+1)]])

		self.candidates = keep_cands

	def tournament_selection4(self):
		# Generate random pairing of candidates (evens face odds)
		samples = list(range(0, len(self.candidates)))

		# Remove the worst-performing candidates if odd numbers
		while len(samples) % 4 != 0:
			samples.pop(np.array([cand.fitness for cand in self.candidates]).argmin())

		# Randomly organize into groups of 4
		shuffle(samples)

		# Define a list of candidates to be retained.
		keep_cands = []
		# For each random quad of candidates, remove less fit of the quad
		# Need to STORE the candidates to be kept since removing  affects
		# the length of the list
		for k in range(0, len(samples), 4):
			fitnesses = np.array([self.candidates[samples[i]].fitness for i in range(k, k+4)])
			winner = samples[k + np.argmax(fitnesses)]
			keep_cands.append(self.candidates[winner])
		self.candidates = keep_cands

	@staticmethod
	def dissimilarity(c1, c2):
		"""
		Calculates how many genes are different between candidates
		"""
		return sum([ex not in c2.examples.values for ex in c1.examples])

	@staticmethod
	def dissim_mat(cands):
		"""
		Calculates a dissimilarity matrix for all candidates
		"""
		output = []
		for c1 in cands:
			dissims = []
			for c2 in cands:
				dissims.append(Population.dissimilarity(c1, c2))
			output.append(dissims)
		return output

	@staticmethod
	def mean_diversity(cands):
		"""
		Calculates the average number of different alleles between candidate pairs
		"""
		arr = np.array(Population.dissim_mat(cands))
		return arr.sum() / (len(cands) * (len(cands) - 1))

	def reproduce(self):
		# Create new offspring, one identical to each parent, to be crossed later
		# Note this only takes even numbers of candidates, which is guaranteed by the previous tournament selection
		new_candidates = []
		for cand in self.candidates:
			new_cand = deepcopy(cand)
			new_cand.fitness = None
			new_candidates.append(new_cand)
		# Generate random pairing of candidates
		shuffle(new_candidates)

		# Perform pmx crossover
		for k in range(0, len(new_candidates), 2):
			self.pmx_crossover(new_candidates[k], new_candidates[k+1])

		# Perform mutation
		for new_cand in new_candidates:
			self.mutate(new_cand)

		# Add new candidates to the population
		self.candidates = self.candidates + new_candidates

	def reproduce_immigrate(self):
		# Create new offspring, one identical to each parent, to be crossed later
		# Note this only takes even numbers of candidates, which is guaranteed by the previous tournament selection
		new_candidates = []
		for cand in self.candidates:
			new_cand = deepcopy(cand)
			new_cand.fitness = None
			new_candidates.append(new_cand)

		# Generate immigrants
		immigrants = []
		for k in range(len(new_candidates)):
			immigrants.append(self.generate_immigrant())
		new_immigrants = deepcopy(immigrants)
		self.candidates = self.candidates + immigrants

		# Perform pmx crossover
		for k in range(0, len(new_candidates)):
			self.pmx_crossover(new_candidates[k], new_immigrants[k])

		# Perform mutation
		for new_cand in new_candidates:
			self.mutate(new_cand)

		# Add new candidates to the population
		self.candidates = self.candidates + new_candidates + new_immigrants


	def pmx_crossover(self, cand1, cand2):
		# Define default indices
		indices = list(range(0,self.n_examples))
		# Choose two random cut points
		cutpt1 = randint(0, self.n_examples-1)
		cutpt2 = randint(0, self.n_examples)
		# ensure cutpt2 > cutpt1; otherwise swap
		if cutpt2 < cutpt1:
			temp = cutpt2
			cutpt2 = cutpt1
			cutpt1 = temp
		# Define the indices of the examples being cut and swapped
		cut = indices[cutpt1 : (cutpt2+1)]

		# Swap selected cuts
		cand1_cut = copy(cand1.examples.iloc[cut])
		cand2_cut = copy(cand2.examples.iloc[cut])
		cand1.examples.iloc[cut] = cand2_cut
		cand2.examples.iloc[cut] = cand1_cut

		cand1_cutsol = copy(cand1.solutions.iloc[cut])
		cand2_cutsol = copy(cand2.solutions.iloc[cut])
		cand1.solutions.iloc[cut] = cand2_cutsol
		cand2.solutions.iloc[cut] = cand1_cutsol

		# Define the partial mapping as a Series
		mapping = pd.Series(data=pd.concat([cand1_cut, cand2_cut]).values, index=pd.concat([cand2_cut, cand1_cut]).values)
		sol_mapping = pd.Series(data=pd.concat([cand1_cutsol, cand2_cutsol]).values, index=pd.concat([cand2_cut, cand1_cut]).values)
		# Perform matching for both candidates
		# BUG: Mapping is switching alleles back

		for i in (list(range(0, cutpt1)) + list(range(cutpt2+1, self.n_examples))):
			# If the allele is contained in the swapped section, map
			while cand1.examples.iloc[i] in cand2_cut.values:
				cand1.solutions.iloc[i] = sol_mapping[cand1.examples.iloc[i]] # must map solution before changing the value
				cand1.examples.iloc[i] = mapping[cand1.examples.iloc[i]]

			while cand2.examples.iloc[i] in cand1_cut.values:
				cand2.solutions.iloc[i] = sol_mapping[cand2.examples.iloc[i]]
				cand2.examples.iloc[i] = mapping[cand2.examples.iloc[i]]

	def mutate(self, cand, generate_new_questions=False):
		# Define the example space without the elements in the candidate being mutated
		if generate_new_questions:
			# Create a new allele using GPT-3 text generation
			for k in range(0, self.n_examples):
				# randomly mutate each allele at the specified rate
				if random() < self.mutation_rate:
					# Create new mutant allele and replace
					cand.examples.iloc[k], cand.solutions.iloc[k] = self.update_train_space()
					cand.mutates += 1

		else:
			# Don't sample an allele which is already in the examples
			valid_example_space = self.example_space[~self.example_space.isin(cand.examples)]
			valid_solution_space = self.solution_space[valid_example_space.index]
			# replace the randomly selected example with another from the valid sample space
			for k in range(0, self.n_examples):
				# randomly mutate each allele at the specified rate
				if random() < self.mutation_rate:
					new_example_series = valid_example_space.sample()
					new_example = new_example_series.iloc[0]
					new_solution = valid_solution_space[new_example_series.index].iloc[0]
					cand.examples[k], cand.solutions[k] = new_example, new_solution
					cand.mutates += 1

	def update_train_space(self, safety_print=False):
		"""
		Calls on davinci GPT-3 to generate new questions for the test space
		"""
		cur_sol = np.random.choice(self.orig_solution_space.unique())
		context_examples = self.orig_example_space[self.orig_solution_space == cur_sol].sample(self.n_mutate_examples)

		# Define the context header for the zero-shot learning to generate new question
		context = self.get_update_context(cur_sol, context_examples)
		response = openai.Completion.create(
			engine="davinci",
			prompt=context,
			temperature=0.8,
			presence_penalty=0.5,
			max_tokens=self.max_mutate_length,
			stop=self.mutate_stop_token,
		)
		result = response["choices"][0]["text"].strip()
		# For safety, print every time we call the API
		if safety_print:
			print("API CALL TO {}: {}".format("davinci", result))

		# Append the new mutation to the sample space and return the result to be placed into new candidate
		self.example_space[self.example_space.size] = result
		self.solution_space[self.solution_space.size] = cur_sol
		return result, cur_sol

	def generate_immigrant(self):
		examples = []
		solutions = []
		for i in range(self.n_examples):
			new_ex, sol = self.update_train_space()
			examples.append(new_ex)
			solutions.append(sol)

		return Candidate(pd.Series(examples), pd.Series(solutions), output_tokens=self.max_tokens, fit_func=self.fit_func, engine=self.engine)

	def get_update_context(self, label, examples):
		context = "Generate a similar question:\n"
		for i, ex in enumerate(examples):
			context += "Q: " + str(ex) + "\n"
		context += "Q:"
		return context

	def to_df(self, k=0):
		output = pd.DataFrame(columns=["gen","id","fitness","mutation","context"])
		for i, candidate in enumerate(self.candidates):
			next_cand = pd.DataFrame({"gen": [k], "id": [i], "fitness": [candidate.fitness], "mutation": [candidate.mutates],
			 "context": [candidate.get_context()]})
			output = pd.concat([output, next_cand], ignore_index=True)
		return output
