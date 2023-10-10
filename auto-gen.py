import tranception
from transformers import PreTrainedTokenizerFast
from tranception import config, model_pytorch
import os
import torch
import argparse
from AR_sampling import estimate_s, compute_k
import math

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, help='Prompt', required=True)
parser.add_argument('--Tmodel', type=str, help='Tranception model path', required=True)
parser.add_argument('--seq_len', type=int, help='Sequence length')
parser.add_argument('--num_samples', type=int, help='Number of samples')
parser.add_argument('--sampling_method', type=str, help='Sampling method')
parser.add_argument('--sampling_threshold', type=float, help='Sampling threshold')
args = parser.parse_args()

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tranception/utils/tokenizers/Basic_tokenizer"),
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

# Define arguments for each sampling method
try:
    threshold = int(args.sampling_threshold)
except TypeError:
    print(f"Invalid sampling threshold: {args.sampling_threshold}")
    threshold = 0

sampling_args = {
    'top_k': {'do_sample': True, 'top_k': threshold},
    'top_p': {'do_sample': True, 'top_p': args.sampling_threshold},
    'greedy': {'do_sample': False},
    'beam_search': {'do_sample': False, 'num_beams': threshold, 'early_stopping': True},
    'random': {'do_sample': True, 'top_k': 0},
    'typical': {'do_sample': True, 'typical_p': args.sampling_threshold},
    'mirostat': {}
}
assert args.sampling_method in sampling_args, f"Invalid sampling method: {args.sampling_method}"

# Load model
model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.Tmodel, local_files_only=True)
model.cuda()
model.config.tokenizer = tokenizer
print("Model successfully loaded from local")

# Initialize prompt
prompt = args.prompt.replace(' ', '').upper().strip()
print(f'Prompt: {prompt}')
print(f'Prompt length: {len(args.prompt)}')
# seq_len = args.seq_len + 2 # +2 for BOS and EOS
AA_to_add = args.seq_len - len(prompt)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print('===============================================')
print(f'Sampling with {args.sampling_method} method and {args.sampling_threshold} threshold')
print('===============================================')

# Generate
for idx in range(args.num_samples):
    if args.sampling_method == 'mirostat':
        target_surprise = args.sampling_threshold
        max_surprise = 2*target_surprise
        error_surprise = 0
        running_tot_surprise = 0
        learning_rate = 1
        num_tokens = AA_to_add
        n=len(tokenizer.vocab)

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
                past = forward.past_key_values

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
        seq = decoded.replace(' ', '')[:args.seq_len]

    else:
        sampling_kwargs = sampling_args[args.sampling_method]
        outputs = model.generate(**inputs, min_new_tokens=AA_to_add, max_new_tokens=AA_to_add, pad_token_id=tokenizer.eos_token_id, 
                          return_dict_in_generate=True, output_scores=True, **sampling_kwargs)
        # Decode for other methods
        decoded = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        seq = decoded[0].replace(' ', '')[:args.seq_len]

    assert len(seq) == args.seq_len, f'Sequence length {len(seq)} does not match {args.seq_len}'
    print(f'Output {idx+1}_{len(seq)}: {seq}')