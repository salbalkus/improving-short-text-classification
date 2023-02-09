from sklearn.metrics import accuracy_score, confusion_matrix
import openai
from random import sample


class Candidate:
	"""
	This class represents a single in-context example set candidate.
	"""
	def __init__(self, examples, solutions, output_tokens=1, fit_func="acc", engine="ada", mutates=0):
		"""
		Args:
			examples: list or pd.Series, with each entry representing an example
			fit_func: str, determines the scoring metric used to calculate fitness
		"""
		self.examples = examples  # List of questions; note all Candidate should have same length list
		self.solutions = solutions
		self.fitness = None  # filler value
		self.output_tokens = output_tokens # Number of tokens desired in output
		self.engine = engine # GPT-3 model to use to generate predictions
		self.mutates = mutates # How many alleles were mutated to generate this candidate?

		# Determine which evaluation function to use to calculate fitness
		if fit_func == "acc":
			self.fit_func = accuracy_score
		else:
			raise Exception("Fitness function not defined.")

	# Call the API to answer an individual question
	# TODO: Currently hardcoded for ">data other" response
	# Use https://beta.openai.com/tokenizer?view=bpe for token codes
	def get_answer(self, q, safety_print=False):
		try:
			response = openai.Completion.create(
				engine=self.engine,
				prompt=self.get_context() + q + "\nTopic:",
				temperature=0,
				#logit_bias={"29": 100,"7890": 100, "584": 100, "1875": 100, "847": 100},
				logit_bias={"6060":100, "3819":100},
				max_tokens=self.output_tokens
			)

			# For safety, print every time we call the API
			if safety_print:
				print("API CALL TO {}: {}".format(self.engine, response["choices"][0]["text"]))

			return str(response["choices"][0]["text"])
		except:
			return "Unknown"

	# Run the algorithm on a set of test questions
	def update_fitness(self, X_test, y_test):
		result = []
		# Construct a context string for GPT-3 from the stored examples
		context = self.get_context()

		X_test_rand = X_test.sample(X_test.size)
		y_test_rand = y_test[X_test_rand.index]
		X_test_rand.reset_index(drop=True)
		y_test_rand.reset_index(drop=True)

		# Iterate through all of the test questions and record the model's answer
		for q in X_test_rand:
			ans = self.translate_result(self.get_answer(q + "\n"))
			result.append(ans)  # Map model output to class and record

		self.test_examples = X_test_rand
		self.true_answers = y_test_rand
		self.test_answers = result

		# Calculate the fitness
		self.fitness = self.fit_func(y_test_rand, result)

	def get_context(self):
		# TODO: This function may change depending on the formatting alleles
		context = 'Decide whether the topic of the question is " Data" or " Other".\n'
		for i, ex in enumerate(self.examples):
			if self.solutions.iloc[i] == "Data":
				context += str(ex) + "\nTopic: Data\n"
			else:
				context += str(ex) + "\nTopic: Other\n"
		return context

	def translate_result(self, result):
		# TODO: This function may change depending on the formatting alleles
		"""
		if result == ">data other":
			return "Data"
		elif result == "data >other":
			return "Other"
		else:
			return "Unknown"
		"""
		return result.strip()

	def print_results(self):
		for i in range(len(self.test_answers)):
			print(self.test_examples[i])
			print("True Answer: " + self.true_answers[i])
			print("Cand Answer: " + self.test_answers[i])

	def confusion(self):
		return confusion_matrix(self.true_answers, self.test_answers)