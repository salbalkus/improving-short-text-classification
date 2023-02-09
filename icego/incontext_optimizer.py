import os
import openai
from dotenv import load_dotenv
from pathlib import Path
from .population import Population
from math import floor
from .cost_optimizer import CostOptimizer
from .candidate import Candidate
import matplotlib.pyplot as plt
import numpy as np

import pickle
import pandas as pd


class InContextOptimizer:
	def __init__(self, path, example_space, solution_space, max_inference_cost=None, n_examples=None, size=8, max_tokens=1, fit_func="acc", engine="ada", mutation_rate=0.1, n_mutate_examples=3, max_mutate_length=64, mutate_stop_token="\n"):
		"""
		Calculates cost of GPT-3 operations and the maximum number of examples to use for a given maximum cost.

		Args:
		:param example_space: list or pd.Series, containing all possible examples to consider as in-context examples
		:param solution_space: list or pd.Series, containing all possible y values associated with each example in example_space
		:param engine: String, defines the GPT-3 model to use for classification (options: "ada", "curie", or "davinci")
		:param max_inference_cost: numeric, the maximum cost in dollars of a single inference call. Higher values yield more accurate models
		:param n_examples: int, number of examples in a given candidate. This represents the size of the example set
		:param size: int, number of candidates in population before reproduction
		:param max_tokens: the expected size of the classification output from GPT-3
		"""

		# Apply the API
		try:
			dotenv_path = Path(path)
			load_dotenv(dotenv_path=dotenv_path)
			openai.api_key = os.environ["OPENAI_API_KEY"]
		except:
			raise Exception("Please specify your OpenAI API key in a .env file as OPENAI_API_KEY=your_key, and provide in the \"path\" parameter")

		self.size = size
		self.max_tokens = max_tokens
		self.fit_func = fit_func
		self.engine = engine

		# Set up the cost optimizer
		self.cost_optimizer = CostOptimizer(engine)
		# Determine how many examples to assign to each candidate in the population
		if max_inference_cost is not None and n_examples is not None:
			raise Exception("Only one of \"max_inference_cost\" and \"n_examples\" may be specified.")

		elif max_inference_cost is not None:
			self.examples_per_candidate = self.cost_optimizer.get_max_examples(example_space, max_inference_cost)[0]

		elif n_examples is not None:
			self.examples_per_candidate = n_examples


		# Construct the population. Note this requires evaluating all candidates.
		self.population = Population(example_space=example_space, solution_space=solution_space,
		                             n_examples=self.examples_per_candidate,
		                             size=size,
		                             max_tokens=max_tokens,
		                             fit_func=self.fit_func,
		                             engine=self.engine,
		                             n_mutate_examples=n_mutate_examples,
		                             max_mutate_length=max_mutate_length,
		                             mutate_stop_token=mutate_stop_token,
		                             mutation_rate=mutation_rate)

		self.population.best = self.population.candidates[0]

		self.df = pd.DataFrame(columns=["gen","id","fitness","mutation","context"])
		self.df = self.df.append(self.population.to_df(0), ignore_index = True)


	def __str__(self):
		return "No. candidates: {}\nNo. examples: {}\nEngine: {}\nBest Fitness: {}\n".format(len(self.population.candidates), self.examples_per_candidate, self.engine, self.population.best.fitness)
		#return str(self.df)

	def train(self, X_test, y_test, epochs=None, max_train_cost=None, online=True, n_mutate_examples=None, n_mutations=None):
		"""
		Optimize in-context examples by evaluating the population of candidates a certain number of times. Retains population for fut
		:param epochs: int; if not None, number of iterations to reproduce and evaluate candidates
		:param max_train_cost: numeric; if not None, iterate candidate evaluation until total cost reaches this value
		:return:
		"""

		# If not using online training, then we should reset the population every time we call the function
		# TODO: If we modify the example space, we will need to change this function to return to the original example space later
		if not online:
			self.population = Population(example_space=self.population.example_space, solution_space=self.population.solution_space, n_examples=self.examples_per_candidate, size=self.population.size, max_tokens=self.max_tokens, fit_func=self.fit_func, engine=self.engine)

		if epochs is not None and max_train_cost is not None:
			raise Exception("Can only specify either epochs or max_train_cost, not both")

		# If we have specified the best cost, calculate the number of epochs available
		elif max_train_cost is not None:
			if n_mutate_examples is not None and n_mutations is not None:
				epochs = floor(self.cost_optimizer.get_max_epochs(sum(self.cost_optimizer.get_token_counts(X_test)), len(X_test), self.size, max_train_cost, n_mutate_examples, n_mutations))
			elif n_mutate_examples is None and n_mutations is None:
				epochs = floor(self.cost_optimizer.get_max_epochs(sum(self.cost_optimizer.get_token_counts(X_test)), len(X_test), self.size, max_train_cost))
			else:
				raise Exception("n_mutate_examples and n_mutations must either both be specified or neither specified.")

		elif epochs is None and max_train_cost is None:
			raise Exception("Must specify either number of epochs or max train cost")

		# Increment from the current epoch if online training
		cur_epoch = 1
		if online:
			cur_epoch = cur_epoch + self.df.gen.max()

		# Evaluate for specified number of epochs
		for i in range(cur_epoch, cur_epoch + epochs):
			print("Epoch {}: Producing next generation...".format(i))
			self.population.next_generation(X_test, y_test)
			self.df = self.df.append(self.population.to_df(i), ignore_index=True)


	def predict(self, input):
		"""
		Obtains a prediction using the best candidate in the population
		:param input: list or str. If list, outputs a list of predictions; if str, outputs a single prediction
		:return:
		"""
		return self.population.best.get_answer(input)

	def generate_test_sample(self):
		"""
		Use GPT-3 to generate additional in-context examples based on existing example space
		:return:
		"""
		pass

	def plot(self, best=False):
		fig, axs = plt.subplots(figsize=(12, 4))
		if best:
			self.df[self.df.id==0].plot.line(ax=axs, x="gen",y="fitness",alpha=0.7)
		else:
			df_plot = self.df.dropna().pivot(index="gen", columns='id', values='fitness')
			df_plot.plot.line(ax=axs, alpha=0.7)
		axs.set_xlabel("Generation")
		axs.set_ylabel("Fitness (" + self.fit_func + ")")
		axs.legend(title='Candidate')

	def save_csv(self, path):
		"""
		Saves the output of the model as a .csv file, for analysis of metrics
		"""
		self.df.to_csv(path, sep="\t")

	def save_pickle(self, path):
		"""
		Saves the model as a pickle.
		:return:
		"""

		pickle_out = open(path, "wb")
		pickle.dump(self, pickle_out)
		pickle_out.close()

	@staticmethod
	def load_pickle(env, model):
		"""
		Load model from a pickle file
		"""

		# Load API Key
		try:
			dotenv_path = Path(env)
			load_dotenv(dotenv_path=dotenv_path)
			openai.api_key = os.environ["OPENAI_API_KEY"]
		except:
			raise Exception("Please specify your OpenAI API key in a .env file as OPENAI_API_KEY=your_key, and provide in the \"path\" parameter")

		pickle_in = open(model, "rb")
		return pickle.load(pickle_in)


