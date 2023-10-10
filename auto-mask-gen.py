import tranception
from transformers import PreTrainedTokenizerFast
from tranception import config, model_pytorch
import os
import torch
import argparse
from AR_sampling import estimate_s, compute_k
import math
from app import replacer, split_mask

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, help='Prompt', required=True)
parser.add_argument('--Tmodel', type=str, help='Tranception model path', required=True)
# parser.add_argument('--seq_len', type=int, help='Sequence length')
parser.add_argument('--mutation_sites', nargs='+', type=int, help='Mutation sites', required=True)
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
if args.sampling_method == 'top_k' or args.sampling_method == 'beam_search':
    assert threshold >= 2, f"{args.sampling_method} requires threshold >= 2"

# Load model
model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.Tmodel, local_files_only=True)
model.cuda()
model.config.tokenizer = tokenizer
print("Model successfully loaded from local")

# Initialize prompt
orig_prompt = args.prompt.replace(' ', '').upper().strip()
prompt = replacer(orig_prompt, args.mutation_sites)
assert '[MASK]' in prompt, 'Prompt must contain [MASK] token'
prompt_parts = split_mask(prompt)
print(f'Prompt: {prompt}')
print(f'Prompt parts: {prompt_parts}')
print(f'Prompt length: {len(orig_prompt)}')
# seq_len = args.seq_len + 2 # +2 for BOS and EOS
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print('===============================================')
print(f'Sampling with {args.sampling_method} method and {args.sampling_threshold} threshold')
print('===============================================')

for idx in range(args.num_samples): # Generate multiple samples
    # Generate text for each prompt part
    generated_texts = ''
    for i, part in enumerate(prompt_parts):
        prompted_text = generated_texts + part
        # print(f'Parts: {part}')
        # print(f'Prompted text: {prompted_text}')
        inputs = tokenizer(prompted_text, return_tensors="pt").to("cuda")
        if part == ' ':
            # Generate
            if args.sampling_method == 'mirostat':
                target_surprise = args.sampling_threshold
                max_surprise = 2*target_surprise
                error_surprise = 0
                running_tot_surprise = 0
                learning_rate = 1
                num_tokens = 1
                n=len(tokenizer.vocab)

                # file_string = args.context
                # f = open(file_string, "r")
                context_text = prompted_text
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
                generated_texts = decoded.replace(' ', '')[:len(prompted_text)+1]

            else:
                sampling_kwargs = sampling_args[args.sampling_method]
                outputs = model.generate(**inputs, min_new_tokens=1, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=True, output_scores=True, **sampling_kwargs)
                # Decode for other methods
                decoded = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                generated_texts = decoded[0].replace(' ', '')[:len(prompted_text)+1]

            # print(f'Generated text: {generated_texts}')
            # print(f'=====================================')
        else:
            generated_texts = prompted_text

    assert len(generated_texts) == len(orig_prompt), f'Sequence length {len(generated_texts)} does not match original {len(orig_prompt)}'
    print(f'Output {idx+1}_{len(generated_texts)}: {generated_texts}')