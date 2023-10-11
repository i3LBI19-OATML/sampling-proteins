import tranception
from transformers import PreTrainedTokenizerFast, XLNetTokenizer, AutoTokenizer, AutoModelForCausalLM, XLNetLMHeadModel
from tranception import config, model_pytorch
import os
import torch
import argparse
from AR_sampling import estimate_s, compute_k
import math
import time
import util
import pandas as pd
from app import process_prompt_protxlnet

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, help='Prompt', required=True)
parser.add_argument('--model_type', type=str, help='Model type', required=True, choices=['ProtXLNet', 'ProtGPT2', 'RITA', 'Tranception'])
parser.add_argument('--local_model', type=str, help='Model path', required=True)
parser.add_argument('--seq_len', type=int, help='Sequence length', required=True)
parser.add_argument('--num_samples', type=int, help='Number of samples, default=1', default=1)
parser.add_argument('--sampling_method', type=str, help='Sampling method', required=True, choices=['top_k', 'top_p', 'greedy', 'beam_search', 'random', 'typical', 'mirostat'])
parser.add_argument('--sampling_threshold', type=float, help='Sampling threshold')

parser.add_argument('--output_name', type=str, help='Output name', required=True)
parser.add_argument('--save_df', action='store_true', help='Save metadata to CSV')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
# Define arguments for each sampling method
if args.sampling_method == 'top_k' or args.sampling_method == 'beam_search':
    assert args.sampling_threshold >= 2, f"{args.sampling_method} requires threshold >= 2"
    threshold = int(args.sampling_threshold)
elif args.sampling_method == 'top_p' or args.sampling_method == 'typical':
    assert args.sampling_threshold <= 1 and args.sampling_threshold > 0, f"{args.sampling_method} requires 0 < threshold <= 1"
    threshold = float(args.sampling_threshold)
else:
    threshold = 0

sampling_args = {
    'top_k': {'do_sample': True, 'top_k': threshold},
    'top_p': {'do_sample': True, 'top_p': threshold},
    'greedy': {'do_sample': False},
    'beam_search': {'do_sample': False, 'num_beams': threshold, 'early_stopping': True},
    'random': {'do_sample': True, 'top_k': 0},
    'typical': {'do_sample': True, 'typical_p': threshold},
    'mirostat': {}
}

# Load model
print(f"=================STARTING=================")
if args.model_type == 'Tranception':
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(args.local_model, local_files_only=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                    unk_token="[UNK]",
                                                    sep_token="[SEP]",
                                                    pad_token="[PAD]",
                                                    cls_token="[CLS]",
                                                    mask_token="[MASK]"
                                                )
elif args.model_type == 'ProtGPT2' or args.model_type == 'RITA':
    tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    model = AutoModelForCausalLM.from_pretrained(args.local_model, local_files_only=True, trust_remote_code=True)
elif args.model_type == 'ProtXLNet':
    tokenizer = XLNetTokenizer.from_pretrained(args.local_model)
    model = XLNetLMHeadModel.from_pretrained(args.local_model)
    
model.cuda()
model.config.tokenizer = tokenizer
print("Model successfully loaded from local")

# Initialize prompt
prompt = args.prompt.replace(' ', '').upper().strip()
prompt = process_prompt_protxlnet(prompt) if args.model_type == 'ProtXLNet' else prompt
print(f'Prompt: {prompt}')
print(f'Prompt length: {len(args.prompt)}')
des_seq_len = args.seq_len + 2 # Add 2 for special tokens
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Initialize list of results
results = []

print('===============================================')
print(f'{args.model_type} with {args.sampling_method} Sampling method and {args.sampling_threshold} threshold')
print('===============================================')
overall_start_time = time.time()

# Generate
for idx in range(args.num_samples):
    start_time = time.time()
    if args.sampling_method == 'mirostat':
        target_surprise = args.sampling_threshold
        max_surprise = 2*target_surprise
        error_surprise = 0
        running_tot_surprise = 0
        learning_rate = 1
        num_tokens = des_seq_len
        n=tokenizer.vocab_size if args.model_type == 'ProtXLNet' else len(tokenizer.vocab)

        # file_string = args.context
        # f = open(file_string, "r")
        context_text = prompt
        context = torch.tensor([tokenizer.encode(context_text)])
        outputs = []
        prev = context
        past = None

        indices_surprise = []

        model.eval()

        # If you have a GPU, put everything on cuda
        context = context.to('cuda')
        model.to('cuda')

        with torch.no_grad():

            for i in range(num_tokens):
                forward = model(input_ids=context, past_key_values=past, return_dict=True)
                logits = forward.logits[0, -1, :]
                past = forward.past_key_values if args.model_type not in ['ProtXLNet', 'RITA'] else None

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

                # Estimate s
                s = estimate_s(prob_original)
                # Compute k
                k = compute_k(n,s,max_surprise)+1

                sorted_logits = sorted_logits[0:k]
                sorted_indices = sorted_indices[0:k]

                prob_topk = torch.softmax(sorted_logits, dim = 0)
                prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
                index_surprise = math.log2(1/prob_original[prev_i])
                indices_surprise.append(index_surprise)

                running_tot_surprise += index_surprise
                prev = sorted_indices[prev_i]
                outputs += prev.tolist()
                context = torch.tensor([prev.tolist()]).to('cuda')

                # adjust max_surprise
                error_surprise = index_surprise - target_surprise
                max_surprise -= learning_rate*error_surprise

        # Decode for mirostat
        decoded = tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded = context_text + decoded # Add prompt to decoded sequence
        seq = decoded.replace(' ', '').replace("\n", "")[:args.seq_len]

    else:
        sampling_kwargs = sampling_args[args.sampling_method]
        outputs = model.generate(**inputs, min_length=des_seq_len, max_length=des_seq_len*2, eos_token_id=2, 
                          return_dict_in_generate=True, output_scores=True, **sampling_kwargs)
        # Decode for other methods
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        seq = decoded[0].replace(' ', '').replace("\n", "")[:args.seq_len]
        # print(f'seq: {seq}')

    assert len(seq) == args.seq_len, f'Sequence length {len(seq)} does not match {args.seq_len}'
    seq_time_taken = round(time.time() - start_time, 3)
    print(f'Output {idx+1}_{len(seq)} {seq_time_taken}s: {seq}') # Print sequence
    # Save results
    samp_thres = None if threshold == 0 else threshold
    name = f'{args.model_type}_{idx+1}_{len(seq)}'
    results.append({'name': name, 'sequence': seq, 'time': seq_time_taken, 'sampling': args.sampling_method, 'threshold': samp_thres})


generated_sequence_df = pd.DataFrame(results)
# Create directory if it doesn't exist
save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ar_protgen")

# Save dataframe to FASTA file
save_path = os.path.join(save_dir, f"{args.output_name}.fasta")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
util.save_as_fasta(generated_sequence_df, save_path)
print(f"FASTA saved to {save_path}")

# Save dataframe to CSV file if requested
if args.save_df:
    save_path = os.path.join(save_dir, f"{args.output_name}.csv")
    generated_sequence_df.to_csv(save_path, index=False)
    print(f"Metadata saved to {save_path}")

overall_time_taken = round(time.time() - overall_start_time, 3)
print(f'===============COMPLETED in {overall_time_taken} seconds=================')