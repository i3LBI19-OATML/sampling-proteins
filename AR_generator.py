import app
import argparse
from transformers import PreTrainedTokenizerFast
import pandas as pd
import os
import util
from AR_sampling import ARtop_k_sampling, ARtemperature_sampler, ARtop_p_sampling, ARtypical_sampling, ARmirostat_sampling, ARrandom_sampling, ARbeam_search
import time
import AR_MCTS

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, help='Sequence to do mutation or DE')
parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], default='small', help='Tranception model size')
parser.add_argument('--Tmodel', type=str, help='Tranception model path')
parser.add_argument('--use_scoring_mirror', action='store_true', help='Whether to score the sequence from both ends')
parser.add_argument('--batch', type=int, default=20, help='Batch size for scoring')
parser.add_argument('--max_pos', type=int, default=50, help='Maximum number of positions per heatmap')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument('--with_heatmap', action='store_true', help='Whether to generate heatmap')
parser.add_argument('--save_scores', action='store_true', help='Whether to save scores')

parser.add_argument('--sampling_method', type=str, choices=['top_k', 'top_p', 'typical', 'mirostat', 'random', 'greedy', 'beam_search', 'mcts'], required=True, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, help='Sampling threshold (k for top_k, p for top_p, tau for mirostat, beam_width in beam_search, etc.)')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for final sampling; 1.0 equals to random sampling')
parser.add_argument('--sequence_num', type=int, required=True, help='Number of sequences to generate')
parser.add_argument('--seq_length', type=int, required=True, help='Length of each sequence to generate')
parser.add_argument('--max_length', type=int, help='Number of search levels in beam search or MCTS')
parser.add_argument('--extension_factor', type=int, default=1, help='Number of AAs to add to extend the sequence in each round')
parser.add_argument('--output_name', type=str, required=True, help='Output file name (Just name with no extension!)')
parser.add_argument('--save_df', action='store_true', help='Whether to save the metadata dataframe')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )
assert args.model or args.Tmodel, "Either model size or model path must be specified"

model = args.model.capitalize()
sequence_num = args.sequence_num
seq_length = args.seq_length
AA_extension = args.extension_factor
generated_sequence = []
sequence_iteration = []
generated_sequence_name = []
mutation_list = []
generation_duration = []
samplings = []
mutants = []
subsamplings = []
samplingtheshold = []
subsamplingtheshold = []

if args.sampling_method in ['top_k', 'top_p', 'typical', 'mirostat', 'beam_search']:
    assert args.sampling_threshold is not None, "Sampling threshold must be specified for top_k, top_p, typical, mirostat, and beam_search sampling methods"
if args.sampling_method == 'beam_search' or args.sampling_method == 'mcts':
    assert args.max_length is not None, "Maximum length must be specified for beam_search or MCTS sampling method"

while len(generated_sequence) < sequence_num:

    if not args.sequence:
        seq = random.choice(AA_vocab)
    else:
        seq = args.sequence.upper()
    sequence_length = len(seq)
    start_time = time.time()
    mutation_history = []
    iteration = 0

    while sequence_length < seq_length:
        print(f"Sequence {len(generated_sequence) + 1} of {sequence_num}, Length {sequence_length} of {seq_length}")
        print("=========================================")

        if args.sampling_method == 'mcts':
            sampling_strat = args.sampling_method
            sampling_threshold = args.max_length
            mutation = AR_MCTS.UCT_search(seq, max_length=args.max_length, model_type=model, tokenizer=tokenizer, AA_vocab=AA_vocab, extension_factor=AA_extension)
            # print("MCTS mutation: ", mutation)
        
        else:
            # Generate possible mutations
            extended_seq = app.extend_sequence_by_n(seq, AA_extension, AA_vocab, output_sequence=True)

            # Score using Tranception (app.score_multi_mutations works for scoring AR sequences)
            scores, _ = app.score_multi_mutations(sequence=None,
                                                        extra_mutants=extended_seq,
                                                        mutation_range_start=None, 
                                                        mutation_range_end=None, 
                                                        model_type=model, 
                                                        scoring_mirror=args.use_scoring_mirror, 
                                                        batch_size_inference=args.batch, 
                                                        max_number_positions_per_heatmap=args.max_pos, 
                                                        num_workers=args.num_workers, 
                                                        AA_vocab=AA_vocab, 
                                                        tokenizer=tokenizer,
                                                        AR_mode=True,
                                                        Tranception_model=args.Tmodel,)

            # Save scores
            if args.save_scores:
                save_path_scores = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_scores.csv")
                scores.to_csv(save_path_scores)
                print(f"Scores saved to {save_path_scores}")

            # 2. Sample mutation from scores
            final_sampler = ARtemperature_sampler(args.temperature)
            sampling_strat = args.sampling_method
            sampling_threshold = args.sampling_threshold

            if sampling_strat == 'top_k':
                assert int(sampling_threshold) <= len(scores), "Top-k sampling threshold must be less than or equal to the number of mutations ({}).".format(len(scores))
                mutation = ARtop_k_sampling(scores, k=int(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'beam_search':
                assert args.max_length < seq_length, "Maximum length must be less than the length of the final sequence"
                mutation = ARbeam_search(scores, beam_width=int(sampling_threshold), max_length=args.max_length, model_type=model, tokenizer=tokenizer, sampler=final_sampler)
            elif sampling_strat == 'top_p':
                assert float(sampling_threshold) <= 1.0 and float(sampling_threshold) > 0, "Top-p sampling threshold must be between 0 and 1"
                mutation = ARtop_p_sampling(scores, p=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'typical':
                assert float(sampling_threshold) < 1.0 and float(sampling_threshold) > 0, "Typical sampling threshold must be between 0 and 1"
                mutation = ARtypical_sampling(scores, mass=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'mirostat':
                mutation = ARmirostat_sampling(scores, tau=float(sampling_threshold), sampler=final_sampler)
            elif sampling_strat == 'random':
                mutation = ARrandom_sampling(scores, sampler=final_sampler)
            elif sampling_strat == 'greedy':
                mutation = ARtop_k_sampling(scores, k=1, sampler=final_sampler)
            else:
                raise ValueError(f"Sampling strategy {sampling_strat} not supported")
            print(f"Using {sampling_strat} sampling strategy with threshold {sampling_threshold}")
            # print("Sampled mutation: ", mutation)

        # 3. Get Mutated Sequence
        mutated_sequence = mutation
        # mutated_sequence = app.get_mutated_protein(seq, mutation) # TODO: Check if this is correct
        if len(mutated_sequence) > seq_length:
            mutated_sequence = mutated_sequence[:seq_length]
        mutation_history += [mutated_sequence.replace(seq, '')]

        print("Original Sequence: ", seq)
        # print("Mutation: ", mutation)
        print("Mutated Sequence: ", mutated_sequence)
        print("=========================================")

        seq = mutated_sequence
        sequence_length = len(seq)
        iteration += 1

    generated_sequence.append(mutated_sequence)
    sequence_iteration.append(iteration)
    samplings.append(sampling_strat)
    samplingtheshold.append(sampling_threshold) 
    seq_name = 'ARTranception_{}AA_{}'.format(iteration+1, len(generated_sequence))
    generated_sequence_name.append(seq_name)
    mutants.append('1')
    subsamplings.append('NA')
    subsamplingtheshold.append('NA')
    mutation_list.append(''.join(mutation_history))
    generation_time = time.time() - start_time
    generation_duration.append(generation_time)
    print(f"Sequence {len(generated_sequence)} of {sequence_num} generated in {generation_time} seconds")
    print("=========================================")
    

generated_sequence_df = pd.DataFrame({'name': generated_sequence_name,'sequence': generated_sequence, 'sampling': samplings, 'threshold': samplingtheshold, 'subsampling':subsamplings, 'subthreshold': subsamplingtheshold, 'iterations': sequence_iteration, 'mutants': mutants, 'mutations': mutation_list, 'time': generation_duration})

if args.save_df:
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ARgenerated_metadata/{}.csv".format(args.output_name))
    os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None
    generated_sequence_df.to_csv(save_path, index=False)
    print(f"Generated sequences saved to {save_path}")

save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ARgenerated_sequence/{}.fasta".format(args.output_name))
os.makedirs(os.path.dirname(os.path.realpath(save_path))) if not os.path.exists(os.path.dirname(os.path.realpath(save_path))) else None
util.save_as_fasta(generated_sequence_df, save_path)
print(f"Generated sequences saved to {save_path}")