'''
Source: https://github.com/brilee/python_uct/blob/master/naive_impl.py
'''
import random
import math
import app
import pandas as pd
import numpy as np

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"

class UCTNode():
  def __init__(self, state, parent=None, prior=0, move=None):
    self.state = state
    self.move = move
    self.is_expanded = False
    self.parent = parent  # Optional[UCTNode]
    self.children = {}  # Dict[move, UCTNode]
    self.prior = prior  # float
    self.total_value = 0  # float
    self.number_visits = 0  # int

  def Q(self):  # returns float
    # print("self.total_value: ", self.total_value, "self.number_visits: ", self.number_visits)
    return self.total_value / (1 + self.number_visits)

  def U(self):  # returns float
    # print("self.parent.number_visits: ", self.parent.number_visits, "self.prior: ", self.prior, "self.number_visits: ", self.number_visits)
    return (math.sqrt(self.parent.number_visits)
        * self.prior / (1 + self.number_visits))

  def best_child(self):
    # for child in self.children.values():
      # print("child: ", child.state, "child.Q(): ", child.Q(), "child.U(): ", child.U(), "child.Q() + child.U(): ", (child.Q() + child.U()))
      # print("=====================================")
    return max(self.children.values(),
               key=lambda node: node.Q() + node.U())

  def select_leaf(self):
    current = self
    while current.is_expanded:
      current = current.best_child()
    return current

  def expand(self, child_priors):
    self.is_expanded = True
    for move, row in child_priors.iterrows():
      self.add_child(move, row['avg_score'], row['mutated_sequence'], row['mutant'])

  def add_child(self, move, prior, child_seq, mutation):
    self.children[move] = UCTNode(
        child_seq, parent=self, prior=prior, move=mutation)

  def backup(self, value_estimate: float):
    current = self
    while current.parent is not None:
      current.number_visits += 1
      current.total_value += value_estimate
      current = current.parent
    print("========END OF ITERATION========")

def UCT_search(state, max_length, extra, model_type, tokenizer, AA_vocab=AA_vocab):
  root = UCTNode(state)
  for _ in range(max_length):
    leaf = root.select_leaf()
    child_priors, value_estimate = Evaluate(leaf.state, extra, model_type, tokenizer, AA_vocab, max_length)
    leaf.expand(child_priors)
    leaf.backup(value_estimate)
    output = max(root.children.items(), key=lambda item: item[1].number_visits)
  return output[1].move

def Evaluate(seq, extra, model_type, tokenizer, AA_vocab, max_length):
    # df_seq = pd.DataFrame.from_dict({'mutated_sequence': [seq]})
    score_heatmap, suggested_mutation, results, _ = app.score_and_create_matrix_all_singles(seq, None, None, model_type, scoring_mirror=False, batch_size_inference=20, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, with_heatmap=False)
    
    results = results.sort_values(by=['avg_score'], ascending=False, ignore_index=True).head(max_length*2)
    extension = app.generate_n_extra_mutations(results, extra, AA_vocab)
    prior, _ = app.score_multi_mutations(sequence=None, extra_mutants=extension, mutation_range_start=None, mutation_range_end=None, model_type=model_type, scoring_mirror=False, batch_size_inference=20, max_number_positions_per_heatmap=50, num_workers=8, AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=True)
    
    child_priors = prior
    value_estimate = float(results['avg_score'].values[0])
    
    return child_priors, value_estimate

