from gpt2_.utils import get_device, get_model, get_tokenizer, free_memory
import numpy as np
import os
import torch
from tqdm import tqdm
from utils.text_processing import dict_to_string


def generate(model, tokenizer, gen_args):
    """Generates a single batch of text.

    Args:
      model: (transformers.GPT2LMHeadModel) The model that will generate text.
      tokenizer: (transformers.GPT2Tokenizer) Tokenizer that splits strings 
        into tokens.
      gen_args: (dict) A subset of args from
        https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
        Also see configs/gpt2_sample.yaml

    Returns:
      gen_strings: (List) Each element is a seperate generated text as a string.
    """
    gen_tensor = model.generate(**gen_args)
    gen_strings = []
    for seq in gen_tensor:
        dec_seq = tokenizer.decode(seq, skip_special_tokens=True)
        gen_strings.append(dec_seq)

    return gen_strings


def randomly_generate(
        checkpoint_dir, target_file, n_batches, batch_size,
        max_len=300, prefix_list=[], show_args=False):
    """Used to generate a lot of text with random sampling params.

    Args:
      checkpoint_dir: (str) Path to a GPT-2 checkpoint.
      target_file: (str) Path to a file where the generated text will be stored.
      n_batches: (int) In total, n_batches * batch_size sequences will be
        generated.
      batch_size: (int) How many sequences to generate per prefix.
      max_len: (int) Maximum allowed length of a generated sequence.
      prefix_list: (list) For each string element, num_return_sequences will be 
        generated. If len(prefix_list) < n_batches, prefix_list will be padded
        with None and the remaining sequences will be generated without a specified 
        prefix
      show_args: (bool) If True, for each generated batch, its randomly generated
        params will also be stored inside target_file.
    """
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(
            'The provided checkpoint directory does not exist.')

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    if os.path.isfile(target_file):
        os.remove(target_file)

    device = get_device()
    model = get_model(checkpoint_dir, None).to(device).eval()
    tokenizer = get_tokenizer()
    free_memory()
    # Delimiters used to separate generated sequences inside target_file
    delim_local = '\n' + '=' * 80 + '\n'
    delim_global = '\n' * 3 + ('#' * 80 + '\n') * 3 + '\n' * 3

    if n_batches > len(prefix_list):
        prefix_list += [None] * (n_batches - len(prefix_list))
    else:
        prefix_list = prefix_list[:n_batches]

    for prefix in tqdm(prefix_list):
        if prefix:
            prefix = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0)
            prefix = prefix.to(device)

        # Randomly generate the arguments used for text generation
        gen_args = {
            'input_ids': prefix,
            'max_length': max_len,
            'do_sample': True,
            'temperature': np.random.uniform(0.7, 1.2),
            'top_k': np.random.randint(20, 80),
            'top_p': np.random.uniform(0.88, 0.97),
            'repetition_penalty': 0.7 if np.random.uniform() > 0.8 else None,
            'num_return_sequences': batch_size,
        }

        generated = generate(model, tokenizer, gen_args)
        generated = delim_local.join(generated)
        if show_args:
            generated = dict_to_string(gen_args) + '\n' + generated
        generated += delim_global
        # Write the generated batch at the end of the file
        with open(target_file, 'a', encoding='utf-8') as file:
            file.write(generated)


def generate_from_config(config):
    """Generates text given a config dict.

    config contains a subset of args from
    https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate

    Args:
      config: A yaml config dict created by gpt2_sample.parse_args()
    """
    checkpoint_dir = config['other_args']['checkpoint_dir']
    target_file = config['other_args']['target_file']
    device = get_device()
    model = get_model(checkpoint_dir, None).to(device).eval()
    tokenizer = get_tokenizer()
    free_memory()
    delim_local = '\n' + '=' * 80 + '\n'
    delim_global = '\n' * 3 + ('#' * 80 + '\n') * 3 + '\n' * 3

    # The last element is an empty string and should be removed
    prefix_list = config['prefixes'].split('\n')[:-1]
    n_batches = config['other_args']['n_batches']
    if n_batches > len(prefix_list):
        prefix_list += [None] * (n_batches - len(prefix_list))
    else:
        prefix_list = prefix_list[:n_batches]

    for prefix in tqdm(prefix_list):
        if prefix:
            prefix = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0)
            prefix = prefix.to(device)

        config['gen_args']['input_ids'] = prefix
        generated = generate(model, tokenizer, config['gen_args'])
        generated = delim_local.join(generated)
        generated += delim_global

        with open(target_file, 'a', encoding='utf-8') as file:
            file.write(generated)
