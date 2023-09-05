# Alignment-based metrics
# ESM-MSA, Identity to closest reference, Subtitution matix (BLOSUM62 or PFASUM15) score mean of mutated positions

# ESM_MSA = True
# Substitution_matrix_score_mean_of_mutated_positions = True
# Identity_to_closest_reference = True

from .util import add_metric, identify_mutation, extract_mutations
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pgen.utils import parse_fasta
from scipy.spatial.distance import pdist
import os
from EVmutation.model import CouplingsModel
from Bio.SeqIO.FastaIO import SimpleFastaParser


# ESM-MSA
def ESM_MSA(target_seqs_file, reference_seqs_file, results):
  print("Scoring with ESM-MSA")
  with tempfile.TemporaryDirectory() as output_dir:
    outfile = output_dir + "/esm_results.tsv"
    try:
      proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm_msa.py"), "-i", target_seqs_file, "-o", outfile, "--reference_msa", reference_seqs_file, "--subset_strategy", "top_hits", "--alignment_size", "31", "--count_gaps", "--mask_distance", "6", "--device", "gpu"], check=True, capture_output=True) # stdout=subprocess.PIPE, stderr=subprocess.PIPE
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e
    # print(proc.stdout)
    # print(proc.stderr)
    df = pd.read_table(outfile)
    for i, row in df.iterrows():
      add_metric(results, row["id"], "ESM-MSA", row["esm-msa"])
    del df

# substitution score
def substitution_score(target_seqs_file, reference_seqs_file, substitution_matrix:str, Substitution_matrix_score_mean_of_mutated_positions:bool, Identity_to_closest_reference:bool, results, gap_open:int = 10, gap_extend:int = 2,):
  #SEARCH #/tmp/mat.mat
  assert substitution_matrix in ["BLOSUM62", "PFASUM15"], "substitution_matrix must be 'BLOSUM62' or 'PFASUM15'"
  search_results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"tmp/ggsearch_results_{substitution_matrix}.txt")
  substitution_matrix_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'tmp/{substitution_matrix}.mat')
  
  with open(search_results_file,"w") as fh:
    try:
      proc = subprocess.run(['ggsearch36', '-f', str(gap_open), '-g', str(gap_extend), '-s', substitution_matrix_file, '-b,' '1', target_seqs_file, reference_seqs_file], check=True, capture_output=True) # stdout=subprocess.PIPE, stderr=subprocess.PIPE
    except subprocess.CalledProcessError as e:
      print(e.stderr.decode('utf-8'))
      print(e.stdout.decode('utf-8'))
      raise e
    # print(proc.stdout.decode('utf-8'), file=fh)
  df = pd.read_csv(substitution_matrix_file, delimiter=r"\s+")
  blosum62 = {}
  for aa1 in df.columns:
    for aa2 in df.index:
      blosum62[(aa1, aa2)] = df.loc[aa1, aa2]
  i = np.random.randint(100000)
  n_train, s_train = parse_fasta(reference_seqs_file,return_names=True)
  n_query,s_query = parse_fasta(target_seqs_file,return_names=True)
  n_query = [nq.strip() for nq in n_query]
  train_seqs = {nt: st for st, nt in zip(s_train,n_train)}
  query_seqs = {nq: sq for sq, nq in zip(s_query,n_query)}

  # with open(output[0],'w') as f: # ell is the number of non-gaps in the query sequence.
      # f.write(f'id,{substitution_matrix},{substitution_matrix}_n_aligned,{substitution_matrix}_ell,{substitution_matrix}_mut_mean,{substitution_matrix}_n_mut,{substitution_matrix}_worst,closest\n')

  with open(search_results_file) as f:
    lines = f.readlines()
  train_coming = False
  qns = [] #query names
  tns = [] #target names
  for i, line in enumerate(lines):
    if '!! No sequences' in line:
      tns.append(None)

    if not train_coming:
      if 'The best scores are:' in line:
        train_coming = True
    else:
      tns.append(line.split()[0])
      train_coming = False
    if 'Library: ' in line:
      qns.append(lines[i - 1].split('>')[-1].split()[0])
  for qn, tn in zip(qns,tns):
    if tn is None: #No hits found
      average_score = str(0) #TODO: I'm not sure what sensible defaults for any of these are
      n_aligned = 0
      mutant_score = 0
      worst_score = 0
      ell = sum([s != '-' for s in query_seqs[qn]]) # just the length of the query sequence.
      tn=""
      identity = 0
    else:
      # write a temporary fasta file
      with tempfile.TemporaryDirectory() as output_dir:

        pairwise_target_fasta = output_dir + '/a_sequence.fasta'
        pairwise_query_fasta = output_dir + '/b_sequence.fasta'
        pairwise_result_file = output_dir + '/pairwise_result.fasta'
        with open(pairwise_target_fasta,'w') as f:
          f.write('>' + tn + '\n' + train_seqs[tn] + '\n')
        with open(pairwise_query_fasta,'w') as f:
          f.write('>' + qn + '\n' + query_seqs[qn] + '\n')
        subprocess.run(["needle", "-gapopen", str(gap_open), "-gapextend", str(gap_extend), "-datafile", substitution_matrix_file, pairwise_target_fasta, pairwise_query_fasta, '-aformat', 'fasta', '-outfile', pairwise_result_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # score it
        seqs = parse_fasta(pairwise_result_file)
        global_score = 0
        mutant_score = 0
        worst_score = np.inf
        n_mutants = 0
        n_aligned = 0
        # print("Closest sequence:", seqs[0])
        seqs_array = np.array([[ord(ss) for ss in list(s)] for s in seqs])
        dist = pdist(seqs_array, metric='hamming')[0]
        identity = 1.0 - dist
        for aa1, aa2 in zip(*seqs):
          if '-' in aa1 + aa2:
            continue
          score = blosum62[(aa1, aa2)]
          global_score += score
          n_aligned += 1
          if aa1 != aa2:
            mutant_score += score
            n_mutants += 1
            if score < worst_score:
                worst_score = score
        if n_mutants > 0:
          mutant_score /= n_mutants
        ell = sum([s != '-' for s in seqs[1]]) # ell is the number of non-gaps in the query sequence.
        if n_aligned == 0: #TODO: same here, are these defaults sensible?
          mutant_score = 0
          worst_score = 0
          average_score = str(0)
          tn=""
        else:
          average_score = str(global_score / n_aligned)
    add_metric(results, qn, f"Closest training sequence ({substitution_matrix})", tn)
    if Substitution_matrix_score_mean_of_mutated_positions:
      add_metric(results, qn, substitution_matrix, mutant_score)
    if Identity_to_closest_reference:      
      add_metric(results, qn, "Identity", identity)

def EVmutation(target_files, orig_seq, results, model_params):
  # Load Model
  c = CouplingsModel(model_params)
  # Load targets
  for targets_fasta in target_files:
    with open(targets_fasta) as fasta_file:  # Will close handle cleanly
      identifiers = []
      seqeunce = []
      for title, sequence in SimpleFastaParser(fasta_file):
          identifiers.append(title.split(None, 1)[0])  # First word is ID
          seqeunce.append(sequence)
      targets = pd.DataFrame({"id":identifiers,"mutated_sequence":seqeunce})
  # Calculate delta E (log-likehood of mutation)
  for i, row in targets.iterrows():
    target = row['mutated_sequence']
    mutations = identify_mutation(orig_seq, target)
    extract = extract_mutations(mutations)
    delta_E, delta_E_couplings, delta_E_fields = c.delta_hamiltonian(extract)
    add_metric(results, row['id'], "EVmutation", delta_E)
  


    