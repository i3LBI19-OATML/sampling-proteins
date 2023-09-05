# Scoring Metrics

Calculation for various sequence- and structure-based quality scores for proteins, such as those produced by generative models.

1. Structue metrics
2. Single-sequence metrics
3. Alignment-based metrics

## 1. Structure metrics
The proteinMPNN and ESM-IF scores are the average log likelihood of the query residues.

### **ProteinMPNN**[^5]
The ProteinMPNN score calculation is performed using the --score_only option of the protein_mpnn_run.py script from the ProteinMPNN repository (https://github.com/dauparas/ProteinMPNN). The ProteinMPNN score is multiplied by -1 so that higher is better, like with other metrics. 

### **ESM-IF**[^3]
The ESM-IF score is calculated using the esm.inverse_folding.util.score_sequence function from the ESM repository (https://github.com/facebookresearch/esm). 
 
### **MIF-ST**[^6]
The MIF-ST score is calculated using the extract_mif.py script from the Protein Sequence Models repository (https://github.com/microsoft/protein-sequence-models).

### **AlphaFold2 pLDDT**
Average taken from the b-factors in AlphaFold2-produced pdb files.

## 2. Single-sequence metrics
### **ESM-1v, ESM-1v mask6, CARP-640m[^7]**
Scores calculated from the ESM-1v and CARP-640M models are the average of the log probabilities of the amino acid in each position. Without masking, this calculation can be done with a single forward pass over each sequence. With partial masking, it can be done in a number of passes equal to 1 / masked_fraction. At the most extreme each position could be masked one at a time. We found that masking in six passes, with the masked positions at a regular interval that shifts on every pass gave scores nearly equivalent to masking one position at a time. Furthermore, when no masking was applied, the scores were shifted towards zero but still strongly correlated with the masked scores. Therefore, we used ESM-1v and CARP-640M scores calculated in one pass without any masking.

### **Repeat 1, 2, 3, 4**
Scores are calculated for the longest single amino acid repeat, the longest 2-mer, 3-mer, and 4-mer repeat in each sequence. The score is calculated as -1 * the number of repeat units. So the sequence MAAAAAAI has a single amino acid repeat score of -6, a 2-mer score of -3, a 3-mer score of -2, and a 4-mer score of -1.

## 3. Alignment-based metrics
### **ESM-MSA**[^4]
Calculated by using pHMMER to find the 31 closest training sequences to each query, aligning the 32 sequences with MAFFT, and calculating the average log probabilities from six passes with a masking interval of six.

### **Identity to closest reference**
The most similar training sequence was found using ggsearch36 from the FASTA3 package, the BLOSUM62 scoring matrix, a gap open penalty of 10 and a gap extend penalty of 2. The hamming distance is then calculated from the gapped alignment between the query and the top hit sequences. Identity is calculated as 1 - hamming_distance.

### **Substitution matrix (_BLOSUM62_[^1] or _PFASUM15_[^2]) score mean of mutated positions**
The closest training sequence was found by FASTA3 ggsearch36. From the alignment to the closest training sequence, the mean BLOSUM62 score across all mismatched positions was calculated, ignoring positions where either query or reference had a gap.

### References
[^1]: Henikoff, S, and J G Henikoff. “Amino Acid Substitution Matrices from Protein Blocks.” Proceedings of the National Academy of Sciences of the United States of America 89, no. 22 (November 15, 1992): 10915–19.

[^2]: Keul, Frank, Martin Hess, Michael Goesele, and Kay Hamacher. “PFASUM: A Substitution Matrix from Pfam Structural Alignments.” BMC Bioinformatics 18, no. 1 (June 5, 2017): 293. https://doi.org/10.1186/s12859-017-1703-z.

[^3]: Hsu, Chloe, Robert Verkuil, Jason Liu, Zeming Lin, Brian Hie, Tom Sercu, Adam Lerer, and Alexander Rives. “Learning Inverse Folding from Millions of Predicted Structures.” bioRxiv, April 10, 2022. https://doi.org/10.1101/2022.04.10.487779.

[^4]: Rao, Roshan M., Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter Abbeel, Tom Sercu, and Alexander Rives. “MSA Transformer.” BioRxiv, February 13, 2021, 2021.02.12.430858. https://doi.org/10.1101/2021.02.12.430858.

[^5]: Dauparas, Justas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, et al. “Robust Deep Learning Based Protein Sequence Design Using ProteinMPNN.” bioRxiv, June 4, 2022. https://doi.org/10.1101/2022.06.03.494563.

[^6]: Yang, Kevin K., Hugh Yeh, and Niccolò Zanichelli. “Masked Inverse Folding with Sequence Transfer for Protein Representation Learning.” bioRxiv, May 28, 2022. https://doi.org/10.1101/2022.05.25.493516.

[^7]: Yang, Kevin K., Alex X. Lu, and Nicolo K. Fusi. “Convolutions Are Competitive with Transformers for Protein Sequence Pretraining.” bioRxiv, May 20, 2022. https://doi.org/10.1101/2022.05.19.492714.

[^8]: 
[^9]: 