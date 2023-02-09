from transformers import GPT2Tokenizer
from math import factorial


class CostOptimizer:
	price_per_token = {
		"ada": 0.0000008,
		"curie": 0.000006,
		"davinci": 0.00006
	}
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	def __init__(self, engine="ada"):
		"""
		Args:
		:param engine: String, defines the GPT-3 model to use for classification (options: "ada", "curie", or "davinci")
		"""
		self.engine = engine

	@staticmethod
	def get_token_counts(example_set):
		return [len(CostOptimizer.tokenizer.tokenize(example)) + 1 for example in example_set]

	def get_max_examples(self, example_space, max_inference_cost=0.0002):
		# We calculate the maximum possible token count for example sets of different sizes by using a greedy algorithm
		# First, calculate the number of tokens in each question (+1 to account for newline)
		token_counts = self.get_token_counts(example_space)
		token_counts.sort(reverse=True)
		# Next, calculate the maximum set size by selecting the largest examples greedily until max price reached
		inference_cost = 0
		max_examples = 0
		for token_count in token_counts:
			new_inference_cost = inference_cost + (token_count * CostOptimizer.price_per_token[self.engine])
			if new_inference_cost > max_inference_cost:
				break
			else:
				inference_cost = new_inference_cost
				max_examples += 1

		return max_examples, inference_cost

	def get_max_epochs(self, test_token_count, test_space_size, n_candidates, max_train_cost, n_davinci_examples=None, mutations_per_epoch=None):
		output = max_train_cost / (test_token_count * self.price_per_token[self.engine] * test_space_size * factorial(n_candidates - 1))

		if n_davinci_examples is not None and mutations_per_epoch is not None:
			output += test_token_count * self.price_per_token["davinci"] * n_davinci_examples * mutations_per_epoch
		elif n_davinci_examples is not None or mutations_per_epoch is not None:
			raise Exception("If davinci mutation is used, both n_davinci_examples and mutations_per_epoch must be defined")

		return output



