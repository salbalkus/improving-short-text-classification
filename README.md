# improving-short-text-classification
A repository to store code and computational results for the paper "Improving Short Text Classification With Augmented Data Using GPT-3" by Salvador Balkus and Donghui Yan.

## Data

The `research_data` directory contains the raw question file collected from the University of Massachusetts Dartmouth Big Data Club Discord Server, as well as tables of questions annotator by each reviewer. Those with names followed by (1), (2), and (3) are the training, validation, and test sets used for training the original model, while those followed by reviewer# measure inter-annotator agreement only. Names are redacted for privacy. 

## Code

`icego` (which stands for In-Context Example Genetic Optimizer) contains code implementing the genetic algorithm as a Python module. It contains the functions to set up the genetic algorithm, evaluate accuracy of candidates, iterate through a specific number of generators, et cetera. The main model is implemented by `incontext_optimizer.py`, which draws on classes defined in the `population.py` and `candidate.py` to maintain a population of candidates. `cost_optimizer.py` contains optional helper functions. 

## Notebooks

The Jupyter notebooks at the root of the project directory contain the computational experiments used to evaluate the two algorithms detailed in the paper.

## Model Outputs

The data and model outputs from the Jupyter notebooks are stored in the `saved_models` directory. Figures for the paper are output in the root of the directory.


