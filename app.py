import torch
import transformers
from transformers import PreTrainedTokenizerFast
import tranception
import datasets
from tranception import config, model_pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow import keras
import itertools

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

def create_all_single_mutants(sequence,AA_vocab=AA_vocab,mutation_range_start=None,mutation_range_end=None):
  all_single_mutants= {}
  sequence_list=list(sequence)
  if mutation_range_start is None: mutation_range_start=1
  if mutation_range_end is None: mutation_range_end=len(sequence)
  for position,current_AA in enumerate(sequence[mutation_range_start-1:mutation_range_end]):
    for mutated_AA in AA_vocab:
      if current_AA!=mutated_AA:
        mutated_sequence = sequence_list.copy()
        mutated_sequence[position] = mutated_AA
        all_single_mutants[current_AA+str(position+1)+mutated_AA]="".join(mutated_sequence)
  all_single_mutants = pd.DataFrame.from_dict(all_single_mutants,columns=['mutated_sequence'],orient='index')
  all_single_mutants.reset_index(inplace=True)
  all_single_mutants.columns = ['mutant','mutated_sequence']
  return all_single_mutants

def extend_sequence_by_n(sequence, n: int, reference_vocab, output_sequence=True):
  permut = ["".join(i) for i in itertools.permutations(reference_vocab, n)]
  extend_sequence = [sequence + ext for ext in permut]
  if output_sequence:
    df_es = pd.DataFrame.from_dict({"mutated_sequence": extend_sequence})
    return df_es
  else:
    df_ext = pd.DataFrame.from_dict({"extension": permut})
    return df_ext

def generate_n_extra_mutations(DMS_data: pd.DataFrame, extra_mutations: int, AA_vocab=AA_vocab, mutation_range_start=None, mutation_range_end=None):
    variants = DMS_data
    seq = variants["mutated_sequence"][0]
    assert extra_mutations > 0, "Number of mutations must be greater than 0."
    if mutation_range_start is None: mutation_range_start=1
    if mutation_range_end is None: mutation_range_end=len(seq)
    count = 0
    while count < extra_mutations:
        print(f"Creating extra mutation {count+1}")
        new_variants = []
        for index, variant in variants.iterrows():
            for i in range(mutation_range_start-1, mutation_range_end):
                for aa in AA_vocab:
                    if aa != variant["mutated_sequence"][i]:
                        new_variant = {
                            "mutated_sequence": variant["mutated_sequence"][:i] + aa + variant["mutated_sequence"][i+1:],
                            "mutant": variant["mutant"] + f":{variant['mutated_sequence'][i]}{i+1}{aa}"
                        }
                        new_variants.append(new_variant)
        count += 1
        variants = pd.DataFrame(new_variants)
    return variants[['mutant','mutated_sequence']]

def trim_DMS(DMS_data:pd.DataFrame, sampled_mutants:pd.DataFrame, mutation_rounds:int):
  for mutation in range(2):
    if mutation == 0:
      DMS_data[f'past_mutation'] = DMS_data["mutant"].map(lambda x: ":".join(x.split(":", mutation_rounds-1)[:mutation_rounds-1]))
    else:
      DMS_data[f'current_mutation'] = DMS_data["mutant"].map(lambda x: ":".join(x.split(":", mutation_rounds-1)[mutation_rounds-1:]))
  trimmed_variants = DMS_data[DMS_data[f'past_mutation'].isin(sampled_mutants['mutant'])].reset_index(drop=True)
  return trimmed_variants[['mutant','mutated_sequence']]

def create_scoring_matrix_visual(scores,sequence,image_index=0,mutation_range_start=None,mutation_range_end=None,AA_vocab=AA_vocab,annotate=True,fontsize=20):
  filtered_scores=scores.copy()
  filtered_scores=filtered_scores[filtered_scores.position.isin(range(mutation_range_start,mutation_range_end+1))]
  piv=filtered_scores.pivot(index='position',columns='target_AA',values='avg_score').round(4)
  mutation_range_len = mutation_range_end - mutation_range_start + 1
  fig, ax = plt.subplots(figsize=(50,mutation_range_len))
  scores_dict = {}
  valid_mutant_set=set(filtered_scores.mutant)  
  ax.tick_params(bottom=True, top=True, left=True, right=True)
  ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True)
  if annotate:
    for position in range(mutation_range_start,mutation_range_end+1):
      for target_AA in list(AA_vocab):
        mutant = sequence[position-1]+str(position)+target_AA
        if mutant in valid_mutant_set:
          scores_dict[mutant]= float(filtered_scores.loc[filtered_scores.mutant==mutant,'avg_score'])
        else:
          scores_dict[mutant]=0.0
    labels = (np.asarray(["{} \n {:.4f}".format(symb,value) for symb, value in scores_dict.items() ])).reshape(mutation_range_len,len(AA_vocab))
    heat = sns.heatmap(piv,annot=labels,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax,vmin=np.percentile(scores.avg_score,2),vmax=np.percentile(scores.avg_score,98),\
                cbar_kws={'label': 'Log likelihood ratio (mutant / starting sequence)'},annot_kws={"size": fontsize})
  else:
    heat = sns.heatmap(piv,cmap='RdYlGn',linewidths=0.30,ax=ax,vmin=np.percentile(scores.avg_score,2),vmax=np.percentile(scores.avg_score,98),\
                cbar_kws={'label': 'Log likelihood ratio (mutant / starting sequence)'},annot_kws={"size": fontsize})
  heat.figure.axes[-1].yaxis.label.set_size(fontsize=int(fontsize*1.5))
  heat.figure.axes[-1].yaxis.set_ticklabels(heat.figure.axes[-1].yaxis.get_ticklabels(), fontsize=fontsize)
  heat.set_title("Higher predicted scores (green) imply higher protein fitness",fontsize=fontsize*2, pad=40)
  heat.set_ylabel("Sequence position", fontsize = fontsize*2)
  heat.set_xlabel("Amino Acid mutation", fontsize = fontsize*2)
  yticklabels = [str(pos)+' ('+sequence[pos-1]+')' for pos in range(mutation_range_start,mutation_range_end+1)]
  heat.set_yticklabels(yticklabels)
  heat.set_xticklabels(heat.get_xmajorticklabels(), fontsize = fontsize)
  heat.set_yticklabels(heat.get_ymajorticklabels(), fontsize = fontsize, rotation=0)
  plt.tight_layout()

  # Save output
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scoring_matrix/')
  os.mkdir(save_path) if not os.path.exists(save_path) else None
  image_path = os.path.join(save_path, 'fitness_scoring_substitution_matrix_{}.png'.format(image_index))
  plt.savefig(image_path,dpi=100)
  # plt.show()

  return image_path

def suggest_mutations(scores, multi=False):
  intro_message = "The following mutations may be sensible options to improve fitness: \n\n"
  #Best mutants
  top_mutants=list(scores.sort_values(by=['avg_score'],ascending=False).head(5).mutant)
  top_mutants_fitness=list(scores.sort_values(by=['avg_score'],ascending=False).head(5).avg_score)
  top_mutants_recos = [top_mutant+" ("+str(round(top_mutant_fitness,4))+")" for (top_mutant,top_mutant_fitness) in zip(top_mutants,top_mutants_fitness)]
  # sorted_mutant_df = pd.DataFrame(list(zip(top_mutants, top_mutants_fitness)), columns =['top_mutants', 'top_mutants_score'])
  mutant_recos = "The single mutants with highest predicted fitness are (positive scores indicate fitness increase Vs starting sequence, negative scores indicate fitness decrease):\n {} \n\n".format(", ".join(top_mutants_recos))
  if not multi:
    #Best positions
    positive_scores = scores[scores.avg_score > 0]
    positive_scores_position_avg = positive_scores.groupby(['position']).mean(numeric_only=True)
    top_positions=list(positive_scores_position_avg.sort_values(by=['avg_score'],ascending=False).head(5).index.astype(str))
    position_recos = "The positions with the highest average fitness increase are (only positions with at least one fitness increase are considered):\n {} \n\n".format(", ".join(top_positions))
    return print(intro_message+mutant_recos+position_recos)
  else:
    return print(intro_message+mutant_recos)

def check_valid_mutant(sequence,mutant,AA_vocab=AA_vocab, multi_mutant=False):
  valid = True
  # print("Original Sequence: ", sequence)
  # print("Mutation: ", mutant)
  if multi_mutant:
    mutant_record = {}
    mutants = mutant.split(':')
    for index, mutant in enumerate(mutants):
      try:
        from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
      except:
        AssertionError(f"Invalid mutant {mutant}")
      mutant_record[index] = mutant
      tmp_list = []
      tmp_mutant_record = mutant_record.copy()
      tmp_mutant_record.popitem()
      for key, value in tmp_mutant_record.items():
        if int(value[1:-1]) == position:
          tmp_list.append(key)
      last_index = max(tmp_list) if len(tmp_list) > 0 else None

      # check if range is valid and mutation is in AA_vocab
      assert int(mutant_record[index][1:-1])>=1 and int(mutant_record[index][1:-1])<=len(sequence),f"position {int(mutant_record[index][1:-1])} is out of range"
      assert mutant_record[index][-1] in AA_vocab, f"to_AA {mutant_record[index][-1]} is not in AA_vocab"
      # check if from_AA is consistent with previous mutant
      if last_index is not None:
        assert mutant_record[last_index][-1] == from_AA, f"To_AA ({mutant_record[last_index][-1]} in {last_index+1}) and From_AA ({from_AA} in {index+1}) are not consistent"
      else:
      # elif sequence[int(mutant_record[index][1:-1])-1]!=from_AA:
        # check if from_AA is consistent with sequence
        assert sequence[int(mutant_record[index][1:-1])-1]==from_AA, f"from_AA {from_AA} at {position} is not consistent with AA in sequence {sequence[int(mutant_record[index][1:-1])-1]} at position {int(mutant_record[index][1:-1])}"

  else:
    try:
      from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
    except:
      valid = False
    assert sequence[position-1]==from_AA, f"from_AA {from_AA} is not consistent with sequence AA {sequence[position-1]}"
    assert position>=1 or position<=len(sequence), f"position {position} is out of range"
    assert to_AA in AA_vocab, f"to_AA {to_AA} is not in AA_vocab"
  return valid

def get_mutated_protein(sequence,mutant):
  mutated_sequence = list(sequence)
  multi_mutant=True if len(mutant.split(':'))>1 else False
  if multi_mutant:
    print("multi_mutant detected")
    assert check_valid_mutant(sequence,mutant, multi_mutant=True), "The mutant is not valid"
    for m in mutant.split(':'):
      from_AA, position, to_AA = m[0], int(m[1:-1]), m[-1]
      mutated_sequence[position-1]=to_AA
    return ''.join(mutated_sequence)
  else:
    print("single mutant detected")
    assert check_valid_mutant(sequence,mutant), "The mutant is not valid"
    from_AA, position, to_AA = mutant[0], int(mutant[1:-1]), mutant[-1]
    mutated_sequence[position-1]=to_AA
  return ''.join(mutated_sequence)

def score_and_create_matrix_all_singles(sequence,mutation_range_start=None,mutation_range_end=None,model_type="Small",scoring_mirror=False,batch_size_inference=20,max_number_positions_per_heatmap=50,num_workers=0,AA_vocab=AA_vocab, tokenizer=tokenizer, with_heatmap=True):
  if mutation_range_start is None: mutation_range_start=1
  if mutation_range_end is None: mutation_range_end=len(sequence)
  assert len(sequence) > 0, "no sequence entered"
  assert mutation_range_start <= mutation_range_end, "mutation range is invalid"
  if model_type=="Small":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Small")
  elif model_type=="Medium":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Medium")
  elif model_type=="Large":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Large")
  if torch.cuda.is_available():
    model.cuda()
    print("Inference will take place on GPU")
  else:
    print("Inference will take place on CPU")
  model.config.tokenizer = tokenizer
  all_single_mutants = create_all_single_mutants(sequence,AA_vocab,mutation_range_start,mutation_range_end)
  print("Single variants generated")
  scores = model.score_mutants(DMS_data=all_single_mutants, 
                                    target_seq=sequence, 
                                    scoring_mirror=scoring_mirror, 
                                    batch_size_inference=batch_size_inference,  
                                    num_workers=num_workers, 
                                    indel_mode=False
                                    )
  print("Single scores computed")
  scores = pd.merge(scores,all_single_mutants,on="mutated_sequence",how="left")
  scores["position"]=scores["mutant"].map(lambda x: int(x[1:-1]))
  scores["target_AA"] = scores["mutant"].map(lambda x: x[-1])
  score_heatmaps = []
  if with_heatmap:
    mutation_range = mutation_range_end - mutation_range_start + 1
    number_heatmaps = int((mutation_range - 1) / max_number_positions_per_heatmap) + 1
    image_index = 0
    window_start = mutation_range_start
    window_end = min(mutation_range_end,mutation_range_start+max_number_positions_per_heatmap-1)
    for image_index in range(number_heatmaps):
      score_heatmaps.append(create_scoring_matrix_visual(scores,sequence,image_index,window_start,window_end,AA_vocab))
      window_start += max_number_positions_per_heatmap
      window_end = min(mutation_range_end,window_start+max_number_positions_per_heatmap-1)
  return score_heatmaps, suggest_mutations(scores), scores, all_single_mutants

def score_multi_mutations(sequence:str, extra_mutants:pd.DataFrame, mutation_range_start=None,mutation_range_end=None,model_type="Small",scoring_mirror=False,batch_size_inference=20,max_number_positions_per_heatmap=50,num_workers=0,AA_vocab=AA_vocab, tokenizer=tokenizer, AR_mode=False, Tranception_model="./Tranception"):
  if sequence is not None:
    if mutation_range_start is None: mutation_range_start=1
    if mutation_range_end is None: mutation_range_end=len(sequence)
  # assert len(sequence) > 0, "no sequence entered"
    assert mutation_range_start <= mutation_range_end, "mutation range is invalid"
  try:
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=Tranception_model, local_files_only=True)
    print("Model successfully loaded from local")
  except:
    print("Model not found locally, downloading from HuggingFace")
    if model_type=="Small":
      model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Small")
    elif model_type=="Medium":
      model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Medium")
    elif model_type=="Large":
      model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Large")
  if torch.cuda.is_available():
    model.cuda()
    print("Inference will take place on GPU")
  else:
    print("Inference will take place on CPU")
  model.config.tokenizer = tokenizer
  scores = model.score_mutants(DMS_data=extra_mutants, 
                                    target_seq=sequence, 
                                    scoring_mirror=scoring_mirror, 
                                    batch_size_inference=batch_size_inference,  
                                    num_workers=num_workers, 
                                    indel_mode=False
                                    )
  print("Scoring done")
  scores = pd.merge(scores,extra_mutants,on="mutated_sequence",how="left")
  # scores["position"]=scores["mutant"].map(lambda x: int(x[1:-1]))
  # scores["target_AA"] = scores["mutant"].map(lambda x: x[-1])
  if AR_mode:
    return scores, extra_mutants
  else:
    return suggest_mutations(scores, multi=True), scores, extra_mutants

def extract_sequence(example):
  label, taxon, sequence = example
  return sequence

def clear_inputs(protein_sequence_input,mutation_range_start,mutation_range_end):
  protein_sequence_input = ""
  mutation_range_start = None
  mutation_range_end = None
  return protein_sequence_input,mutation_range_start,mutation_range_end, extra_mutants

def predict_proteinBERT(model, DMS, input_encoder, top_n, batch_size):
  seqs = DMS["mutated_sequence"]
  seq_len = 512

  X = input_encoder.encode_X(seqs, seq_len)
  y_pred = model.predict(X, batch_size = batch_size)

  DMS['prediction'] = y_pred
  DMS = DMS.sort_values(by = 'prediction', ascending = False, ignore_index = True)
  DMS = DMS.head(top_n)
  return DMS[['mutated_sequence', 'mutant']]

def load_savedmodel(model_path):
  model = keras.models.load_model(model_path)
  return model

