# Structure metrics
# ESM-IF, ProteinMPNN, MIF-ST, AlphaFold2 pLDDT

# ESM_IF = True 
# ProteinMPNN = True 
# MIF_ST = True
# AlphaFold2_pLDDT = True

from .util import add_metric, get_pdb_sequence, residues_in_pdb
import esm
from glob import glob
from pathlib import Path
import tempfile
import subprocess
import pandas as pd
from biotite.structure.io import pdb
import os
import tmscoring

# ESM-IF
def ESM_IF(pdb_files, results): #TODO: move to GPU? Maybe spin off into a subprocess when moving to GPU, to avoid memory leaks?
  esm_if_model, esm_if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
  # esm_if_model = esm_if_model.to(device) # TODO: for some reason it crashes when I move it to the GPU
  esm_if_model.eval()
        # with open(output[0],'w') as f:
        #     f.write('id,esm-if\n')
  for pdb_file in pdb_files:
    fstem = Path(pdb_file).stem
    name = fstem
    coords, seq = esm.inverse_folding.util.load_coords(pdb_file, "A")
    ll, _ = esm.inverse_folding.util.score_sequence(
    esm_if_model, esm_if_alphabet, coords, str(seq))
    add_metric(results, name, "ESM-IF", ll)
  del esm_if_model
  del esm_if_alphabet

# ProteinMPNN
def ProteinMPNN(pdb_files, results):
  with tempfile.TemporaryDirectory() as output_dir:
    for i, pdb_file in enumerate(pdb_files):
      command_line_arguments=[
      "python",
      os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProteinMPNN/vanilla_proteinmpnn/protein_mpnn_run.py"),
      "--pdb_path", pdb_file,
      "--pdb_path_chains", "A",
      "--score_only", "1",
      "--save_score", "1",
      "--out_folder", output_dir,
      "--batch_size", "1"
      ]
      fstem = Path(pdb_file).stem
      name = fstem
      outfile = output_dir + f"outfile_{i}.txt"
      with open(outfile,"w") as fh:
        proc = subprocess.run(command_line_arguments, stdout=subprocess.PIPE, check=True)
        print(proc.stdout.decode('utf-8'), file=fh)
      with open(outfile,"r") as score_file_h:
        score_file_lines = score_file_h.readlines()
      score_line = score_file_lines[-2].split(",")
      score_parts = score_line[1].strip().split(": ")
      assert score_parts[0] == "mean" 
      score = -1 * float(score_parts[1]) 
      add_metric(results, name, "ProteinMPNN", score)

# MIF-ST
def MIF_ST(pdb_files, results, device): 
  with tempfile.TemporaryDirectory() as output_dir:
    spec_file_path = output_dir + "/spec_file.tsv"
    with open(spec_file_path, 'w') as f:
      f.write('name\tsequence\tpdb\n')
      for pdb_file in pdb_files:
        seq = get_pdb_sequence(pdb_file) 
        name = Path(pdb_file).stem
        f.write(name + '\t' + seq + '\t' + pdb_file + '\n')
    #print(spec_file_path)
    proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp/extract_mif.py"), "mifst", spec_file_path, output_dir + "/", "logits", "--include", "logp", "--device", device], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(proc.stderr.decode("utf-8"))
    print(proc.stdout.decode("utf-8"))
    df = pd.read_table(output_dir + '/mifst_logp.tsv')
    df = df.rename(columns={'name': 'id', 'logp': 'mifst_logp'}, )
    for _, row in df.iterrows():
      add_metric(results, row["id"], "MIF-ST", row["mifst_logp"])

# pLDDT
def AlphaFold2_pLDDT(pdb_files, results):
  for pdb_file in pdb_files:
    fstem = Path(pdb_file).stem
    name = fstem
    pdb_file = pdb.PDBFile.read(pdb_file)
    atoms = pdb_file.get_structure(extra_fields = ['b_factor'])
    prev_residue = -1
    plddt_sum = 0
    residue_count = 0
    for a in atoms[0]:
      if a.res_id != prev_residue:
        prev_residue = a.res_id
        residue_count += 1
        plddt_sum += a.b_factor
    add_metric(results, name, "AlphaFold2 pLDDT", plddt_sum/residue_count)

def TM_score(pdb_files, reference_pdb, results):
  for pdb_file in pdb_files:
    fstem = Path(pdb_file).stem
    name = fstem
    # PDB1 = Reference; PDB2 = Target
    tmscore = tmscoring.get_tm(reference_pdb, pdb_file)
    add_metric(results, name, "TM Score", tmscore)
