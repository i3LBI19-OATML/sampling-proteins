# Single sequence metrics
# ESM-1v, ESM-1v-mask6, CARP-640m-logp, Repeat-1, Repeat-2, Repeat-3, Repeat-4

# CARP_640m_logp = True
# ESM_1v = True
# ESM_1v_mask6 = True
# repeat_1 = True
# repeat_2 = True
# repeat_3 = True
# repeat_4 = True

from .util import add_metric
import tempfile
import subprocess
import pandas as pd
from glob import glob
import torch
from pgen.utils import parse_fasta
import os
from Bio.SeqIO.FastaIO import SimpleFastaParser

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import model_pytorch


# target_seqs_file = "/tmp/target_seqs.fasta"
# with open(target_seqs_file,"w") as fh:
#   for target_fasta in glob("/target_seqs/*"):
#     for name, seq in zip(*parse_fasta(target_fasta, return_names=True, clean="unalign")):
#       print(f">{name}\n{seq}", file=fh)

#CARP
def CARP_640m_logp(target_seqs_file, results, device): 
  with tempfile.TemporaryDirectory() as output_dir:
    proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp/extract.py"), "carp_640M", target_seqs_file, output_dir + "/", "--repr_layers", "logits", "--include", "logp", "--device", device], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # print(proc.stderr)
    # print(proc.stdout)
    df = pd.read_table(output_dir + '/carp_640M_logp.tsv')
    df = df.rename(columns={'name': 'id', 'logp': 'carp640m_logp'},)
    for _, row in df.iterrows():
      add_metric(results, row["id"], "CARP-640m", row["carp640m_logp"])
    del df

# ESM1v (unmasked)
def ESM_1v(target_files, results, device): #TODO: allow other devices?
  if device=='cuda:0':
    torch.cuda.empty_cache()
  for targets_fasta in target_files:
    with tempfile.TemporaryDirectory() as output_dir:
      outfile = output_dir + "/esm_results.tsv"
      proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm.py"), "-i", targets_fasta, "-o", outfile, "--model", "esm1v", "--masking_off", "--score_name", "score", "--device", "gpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      # print(proc.stdout)
      # print(proc.stderr)
      df = pd.read_table(outfile)
      for i, row in df.iterrows():
        add_metric(results, row["id"], "ESM-1v", row["score"])
      del df

# ESM1v mask 6
def ESM_1v_mask6(target_files, results, device): #TODO: allow other devices?
  if device=='cuda:0':
    torch.cuda.empty_cache()
  for targets_fasta in target_files:
    with tempfile.TemporaryDirectory() as output_dir:
      outfile = output_dir + "/esm_results.tsv"
      proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm.py"), "-i", targets_fasta, "-o", outfile, "--model", "esm1v", "--mask_distance", "6", "--score_name", "score", "--device", "gpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      # print(proc.stdout)
      # print(proc.stderr)
      df = pd.read_table(outfile)
      for i, row in df.iterrows():
        add_metric(results, row["id"], "ESM-1v mask6", row["score"])
      del df

# repeat
def find_longest_repeat(seq, k):
  longest = [1] * len(seq)
  pattern = [None] * len(seq)
  
  seq_len = len(seq)
  for i in range(seq_len):
    if i + k <= seq_len:
      pattern[i] = seq[i:i+k]
    if i - k >= 0:
      if pattern[i-k] == pattern[i]:
        longest[i] = longest[i-k] + 1
  return -1 * max(longest)

def Repeat(target_files, repeat_score, results):
    repeat_ks = list()
    if repeat_score['repeat_1']:
        repeat_ks.append(1)
    if repeat_score['repeat_2']:
        repeat_ks.append(2)
    if repeat_score['repeat_3']:
        repeat_ks.append(3)
    if repeat_score['repeat_4']:
        repeat_ks.append(4)

    for k in repeat_ks:
        for targets_fasta in target_files:
            for name, seq in zip(*parse_fasta(targets_fasta, return_names=True, clean="unalign")):
                score = find_longest_repeat(seq, k)
                add_metric(results, name, f"longest_repeat_{k}", score)

# Tranception
def Tranception(target_files, orig_seq, results, device, model_type="Small"):
  if device=='cuda:0':
    torch.cuda.empty_cache()
  for targets_fasta in target_files:
    with open(targets_fasta) as fasta_file:  # Will close handle cleanly
      identifiers = []
      seqeunce = []
      for title, sequence in SimpleFastaParser(fasta_file):
          identifiers.append(title.split(None, 1)[0])  # First word is ID
          seqeunce.append(sequence)
      targets = pd.DataFrame({"id":identifiers,"mutated_sequence":seqeunce})
    print("Tranception scores computing...")
    if model_type=="Small":
      model = model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Small")
    elif model_type=="Medium":
      model = model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Medium")
    elif model_type=="Large":
      model = model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="PascalNotin/Tranception_Large")
    if torch.cuda.is_available():
      model.cuda()
      print("Inference will take place on GPU")
    else:
      print("Inference will take place on CPU")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizers/Basic_tokenizer"),
                                                    unk_token="[UNK]",
                                                    sep_token="[SEP]",
                                                    pad_token="[PAD]",
                                                    cls_token="[CLS]",
                                                    mask_token="[MASK]"
                                                )
    model.config.tokenizer = tokenizer
    scores = model.score_mutants(DMS_data=targets, 
                                    target_seq=orig_seq, # need template seq
                                    scoring_mirror=False, 
                                    batch_size_inference=20,  
                                    num_workers=0, 
                                    indel_mode=True # since only mutated_sequence is provided
                                    )
    print("Tranception scores computed")
    scores = pd.merge(scores,targets,on="mutated_sequence",how="left")
    for i, row in scores.iterrows():
        add_metric(results, row["id"], "Tranception", row["avg_score"])
    del scores