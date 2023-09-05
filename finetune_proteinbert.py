from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

from tensorflow import keras
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Finetune ProteinBERT on a dataset.')
parser.add_argument('--train_data', type=str, default='avGFP_dataset/fluorescence.train.csv', help='Name of the training set file.')
parser.add_argument('--valid_data', type=str, default='avGFP_dataset/fluorescence.valid.csv', help='Name of the validation set file.')

parser.add_argument('--save_name', type=str, default='avGFP_finetuned_model', help='Name of the finetuned model to be saved.')
args = parser.parse_args()

# Defining the output type and the output specification
OUTPUT_TYPE = OutputType(False, 'numeric')
UNIQUE_LABELS = None
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

# Loading the datasets
train_set_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "{}".format(args.train_data))
valid_set_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "{}".format(args.valid_data))

train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
valid_set = pd.read_csv(valid_set_file_path).dropna().drop_duplicates()

assert 'seq' in train_set.columns and 'label' in train_set.columns, 'Training set must have "seq" and "label" columns.'
assert 'seq' in valid_set.columns and 'label' in valid_set.columns, 'Validation set must have "seq" and "label" columns.'
print(f'{len(train_set)} training set records, {len(valid_set)} validation set records.')


# Loading the pre-trained model and fine-tuning it on the loaded dataset
original_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_bert/original_weights/")
pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir=original_weights, validate_downloading = False)

# get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
        get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
]

finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
        seq_len = 512, batch_size = 32, max_epochs_per_stage = 40, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
        lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-05, callbacks = training_callbacks)

# Saving the fine-tuned model
save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "finetuned_models/{}".format(args.save_name))
os.makedirs(os.path.dirname(os.path.realpath(save_dir))) if not os.path.exists(os.path.dirname(os.path.realpath(save_dir))) else None

model = model_generator.create_model(512)
model.save(save_dir)
print(f'Model saved to {save_dir}.')