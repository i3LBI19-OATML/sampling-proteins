device = 'cuda:0'

from glob import glob
from pgen.utils import parse_fasta
import pandas as pd
import os
import argparse

from scoring_metrics import structure_metrics as st_metrics
from scoring_metrics import single_sequence_metrics as ss_metrics
from scoring_metrics import alignment_based_metrics as ab_metrics
from scoring_metrics import fid_score as fid

#Reset calculated metrics (creates a new datastructure to store results, clearing any existing results)
results = dict()

# Default directories
file_dir = os.path.dirname(os.path.realpath(__file__))
default_pdb_dir = os.path.join(file_dir, "pdbs")
default_reference_pdb_dir = os.path.join(file_dir, "reference_pdbs")
default_reference_dir = os.path.join(file_dir, "reference_seqs")
default_target_dir = os.path.join(file_dir, "target_seqs")

os.makedirs(default_pdb_dir) if not os.path.exists(default_pdb_dir) else None
os.makedirs(default_reference_pdb_dir) if not os.path.exists(default_reference_pdb_dir) else None
os.makedirs(default_reference_dir) if not os.path.exists(default_reference_dir) else None
os.makedirs(default_target_dir) if not os.path.exists(default_target_dir) else None

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pdb_dir", type=str, default=default_pdb_dir, help="Directory containing pdb files")
parser.add_argument("--reference_dir", type=str, default=default_reference_dir, help="Directory containing reference fasta files")
parser.add_argument("--ref_pdb_dir", type=str, default=default_reference_pdb_dir, help="Directory containing reference pdb files")
parser.add_argument("--target_dir", type=str, default=default_target_dir, help="Directory containing target fasta files")
parser.add_argument("--sub_matrix", type=str, choices=["blosum62", "pfasum15"], default="blosum62", help="Substitution matrix to use for alignment-based metrics")
parser.add_argument("--remove_sub_score_mean", action="store_false", help="Whether to not score the mean of the scores for mutated sequences")
parser.add_argument("--remove_identity", action="store_false", help="Whether to not score the identity of the mutated sequence to the closest reference sequence")
parser.add_argument("--sub_gap_open", type=int, default=10, help="Gap open penalty for alignment-based metrics")
parser.add_argument("--sub_gap_extend", type=int, default=2, help="Gap extend penalty for alignment-based metrics")
parser.add_argument("--remove_repeat_score_1", action="store_false", help="Whether to not score the first repeat")
parser.add_argument("--remove_repeat_score_2", action="store_false", help="Whether to not score the second repeat")
parser.add_argument("--remove_repeat_score_3", action="store_false", help="Whether to not score the third repeat")
parser.add_argument("--remove_repeat_score_4", action="store_false", help="Whether to not score the fourth repeat")
parser.add_argument("--score_structure", action="store_true", help="Whether to score structural metrics")
parser.add_argument("--use_tranception", action="store_true", help="Whether to use Tranception")
parser.add_argument("--use_evmutation", action="store_true", help="Whether to use EVmutation")
parser.add_argument("--skip_FID", action="store_true", help="Whether to not calculate FID")
parser.add_argument("--model_params", type=str, help="Model params to use for EVmutation")
parser.add_argument("--orig_seq", type=str, help="Original sequence to use for Tranception or EVmutation")
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
args = parser.parse_args()

score_structure = args.score_structure

# Checks
if args.use_tranception or args.use_evmutation:
  assert args.orig_seq, "Must specify original sequence if using Tranception or EVmutation"
if args.use_evmutation:
  assert args.model_params, "Must specify model params if using EVmutation"
  assert os.path.exists(args.model_params), f"Model params {args.model_params} does not exist"

# Check that the required directories exist
if score_structure:
  pdb_dir = os.path.abspath(args.pdb_dir)
  ref_pdb_dir = os.path.abspath(args.ref_pdb_dir)
reference_dir = os.path.abspath(args.reference_dir)
target_dir = os.path.abspath(args.target_dir)

if score_structure:
  assert os.path.exists(pdb_dir), f"PDB directory {pdb_dir} does not exist"
  assert os.path.exists(ref_pdb_dir), f"Reference PDB directory {ref_pdb_dir} does not exist"
assert os.path.exists(reference_dir), f"Reference directory {reference_dir} does not exist"
assert os.path.exists(target_dir), f"Target directory {target_dir} does not exist"

# Check that the required files exist
if score_structure:
  pdb_files = glob(pdb_dir + "/*.pdb")
  reference_pdb_files = glob(ref_pdb_dir + "/*.pdb")
reference_files = glob(reference_dir + "/*.fasta")
target_files = glob(target_dir + "/*.fasta")

if score_structure:
  assert len(pdb_files) > 0, f"No pdb files found in {pdb_dir}"
  assert len(reference_pdb_files) > 0, f"No reference pdb files found in {ref_pdb_dir}"
assert len(reference_files) > 0, f"No reference fasta files found in {reference_dir}"
assert len(target_files) > 0, f"No target fasta files found in {target_dir}"

if score_structure:
# Structure metrics
# ESM-IF, ProteinMPNN, MIF-ST, AlphaFold2 pLDDT, TM-score
  if len(reference_pdb_files) > 1:
    print("Found multiple reference pdb files, using the first one")
    print(f"Found {len(reference_pdb_files)} reference pdb files, using {reference_pdb_files[0]}")
  reference_pdb = reference_pdb_files[0]

  st_metrics.TM_score(pdb_files, reference_pdb, results)
  st_metrics.ESM_IF(pdb_files, results)
  st_metrics.ProteinMPNN(pdb_files, results)
  st_metrics.MIF_ST(pdb_files, results, device)
  st_metrics.AlphaFold2_pLDDT(pdb_files, results)

# Alignment-based metrics
# ESM-MSA, Identity to closest reference, Subtitution matix (BLOSUM62 or PFASUM15) score mean of mutated positions
# FID (ESM-1v), EVmutation
sub_matrix = args.sub_matrix.upper()
score_mean = args.remove_sub_score_mean
identity = args.remove_identity
sub_gap_open = args.sub_gap_open
sub_gap_extend = args.sub_gap_extend

#concatenate reference sequences
# Reference sequences
reference_seqs_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scoring_metrics/tmp/reference_seqs.fasta")
with open(reference_seqs_file,"w") as fh:
  for reference_fasta in reference_files:
    for name, seq in zip(*parse_fasta(reference_fasta, return_names=True, clean="unalign")):
      print(f">{name}\n{seq}", file=fh)

# Target sequences
target_seqs_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scoring_metrics/tmp/target_seqs.fasta")
with open(target_seqs_file,"w") as fh:
  for target_fasta in target_files:
    for name, seq in zip(*parse_fasta(target_fasta, return_names=True, clean="unalign")):
      print(f">{name}\n{seq}", file=fh)

if not args.skip_FID:
  fretchet_score = fid.calculate_fid_given_paths(target_files, reference_files, device)
else:
  fretchet_score = None

ab_metrics.ESM_MSA(target_seqs_file, reference_seqs_file, results)
try:
  ab_metrics.substitution_score(target_seqs_file, reference_seqs_file,
                                substitution_matrix=sub_matrix, 
                                Substitution_matrix_score_mean_of_mutated_positions=score_mean, 
                                Identity_to_closest_reference=identity,
                                results=results,
                                gap_open=sub_gap_open,
                                gap_extend=sub_gap_extend,)
  identity_score = True
except:
  identity_score = False

if args.use_evmutation:
  ab_metrics.EVmutation(target_files=target_files, orig_seq=args.orig_seq.upper(), results=results, model_params=args.model_params)

# Single sequence metrics
# ESM-1v, ESM-1v-mask6, CARP-640m-logp, Repeat-1, Repeat-2, Repeat-3, Repeat-4, Tranception
target_seqs_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scoring_metrics/tmp/target_seqs.fasta")
with open(target_seqs_file,"w") as fh:
  for target_fasta in target_files:
    for name, seq in zip(*parse_fasta(target_fasta, return_names=True, clean="unalign")):
      print(f">{name}\n{seq}", file=fh)

repeat_score = dict()
repeat_score['repeat_1'] = args.remove_repeat_score_1
repeat_score['repeat_2'] = args.remove_repeat_score_2
repeat_score['repeat_3'] = args.remove_repeat_score_3
repeat_score['repeat_4'] = args.remove_repeat_score_4

ss_metrics.CARP_640m_logp(target_seqs_file, results, device)
ss_metrics.ESM_1v(target_files, results, device)
ss_metrics.ESM_1v_mask6(target_files, results, device)
ss_metrics.Repeat(target_files, repeat_score, results)
if args.use_tranception:
  ss_metrics.Tranception(target_files=target_files, orig_seq=args.orig_seq.upper(), results=results, device=device, model_type="Small")

# Download results
df = pd.DataFrame.from_dict(results, orient="index")
df["FID"] = fretchet_score
if not identity_score:
  df['Identity'] = None

if score_structure:
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "{}.csv".format(args.output_name))
else :
  save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "{}_no_structure.csv".format(args.output_name))
os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None

df.to_csv(save_path)
print(f"Results saved to {save_path}")
print("===========================================")

