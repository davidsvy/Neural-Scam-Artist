import gc
import numpy as np
import random

import torch
from gpt2_.dataset import get_datasets
import transformers


def set_seed(seed_val):
    """Sets seed for reproducibility.

    Args:
      seed_val: (int) Seed for rng.
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    transformers.set_seed(seed_val)


def get_device():
    """Returns Cuda device if it is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'Available GPU: {torch.cuda.get_device_name(0)}.')
    else:
        print('GPU unavailable.')

    return device


def pytorch_bs():
    """Might make training slightly faster.

    Everything that is written on the internet is true.
    """
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)


def get_tokenizer():
    """Returns tokenizer.
    """
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        'gpt2', bos_token='<sos>', eos_token='<eos>', pad_token='<pad>')
    return tokenizer


def get_model(checkpoint_dir, tokenizer):
    """Creates a Huggingface GPT-2 model.

    Args:
      checkpoint_dir: (str) If provided the model weights are loaded from 
        this checkpoint. If given None or False, the model is initialized
        with the pretrained Huggingface weights.
      tokenizer: (transformers.GPT2Tokenizer) Tokenizer that splits strings 
        into tokens.

    Returns: (transformers.GPT2LMHeadModel) Model.
    """
    if checkpoint_dir:
        return transformers.GPT2LMHeadModel.from_pretrained(checkpoint_dir)

    configuration = transformers.GPT2Config.from_pretrained(
        'gpt2', output_hidden_states=False)
    model = transformers.GPT2LMHeadModel.from_pretrained(
        "gpt2", config=configuration)
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_dynamic_padding_collator(pad_idx):
    """Returns a data collator.

    The nested function structure is necassary because the data collator 
    might require additional parameters, but its only input must be the batch.

    Args:
      pad_idx: (int) The index of the padding token. In reality this parameter
        does not affect training as all padding will be masked by the model.

    Returns: 
      dynamic_padding_collator: (function) Function that collects samples from
        the dataset and constucts a batch.
    """

    def dynamic_padding_collator(batch):
        """Dynamically pads a batch, creates mask and masks out padded labels.

        Given batch_size sequences of different lengths, computes
        k = log2(batch_size). All sequences are padded/truncated to the length 
        of the kth longest sequence. Additionally, masks are constructed based 
        on the new dynamic lengths and padded tokens are masked out within labels.
        Code Based on:
        https://gist.github.com/pommedeterresautee/1a334b665710bec9bb65965f662c94c8
        https://huggingface.co/transformers/_modules/transformers/data/data_collator.html#default_data_collator

        Args:
          batch: (list) Each element is a dict {'input_ids': tokens} where tokens
            is a list of token indices coreesponding to a single sample.

        Returns:
          batch: (dict) Contains 3 elements: 
            input_ids: (torch.tensor) Padded input indices.
            attention_mask: (torch.tensor) Mask that takes the value 1 for valid
              tokens and 0 for padded ones.
            labels: (torch.tensor) Same as input_ids but padded tokens are replaced
              with -100.
        """
        batch_size = len(batch)
        lens = torch.Tensor([len(sample['input_ids'])
                             for sample in batch]).long()
        k = max(1, int(np.round(np.log2(batch_size))))
        idx = batch_size - k + 1
        max_len = torch.kthvalue(lens, idx).values
        max_len_ = max_len.item()

        def pad_seq(seq):
            if len(seq) < max_len_:
                return seq + (max_len_ - len(seq)) * [pad_idx]
            return seq[:max_len_]

        input_ids = [pad_seq(sample['input_ids']) for sample in batch]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        lens = torch.minimum(lens, max_len)
        masks = (torch.arange(max_len_)[None, :] < lens[:, None]).long()

        """From https://huggingface.co/transformers/model_doc/gpt2.html:
         Indices are selected in [-100, 0, ..., config.vocab_size - 1] 
         All labels set to -100 are ignored (masked), the loss is only computed 
         for labels in [0, ..., config.vocab_size - 1]

         WTF??? WTF??? WTF??? WTF??? WHO CAME UP WITH THIS??? THIS MAKES NO SENSE.
         WHY -100? WHY NOT -69 OR -420? WHAT IS THE POINT OF THE ATTENTION MASK THEN??? 
         IT IS SUPPOSED TO MASK OUT THE LOSS FOR PADDED TOKENS. HOWEVER, IF 
         LABELS == INPUT_IDS, THE MODEL LEARNS TO REPEAT THE PADDING TOKEN AFTER THE END 
         OF THE TEXT. THIS MEANS THAT THE LABELS MUST BE MANUALLY MASKED.
        """
        # mask might need shifting
        labels = input_ids.clone()
        labels.masked_fill_(torch.logical_not(masks), -100)

        batch = {
            'input_ids': input_ids, 'attention_mask': masks, 'labels': labels}
        return batch

    return dynamic_padding_collator


def free_memory():
    """(Maybe) prevents Cuda running out of memory
    """
    gc.collect()
    torch.cuda.empty_cache()


class Garbage_collector_callback(transformers.TrainerCallback):
    """Custom callback that (maybe) prevents Cuda running out of memory.

    I have absolutely no idea if this actually helps. However, Cuda on Colab
    is prone to memory leaks, especially in case of Ctrl + C interrupts. 
    After using this callback the issue kinda disappeared. Code based on 
    https://huggingface.co/transformers/main_classes/callback.html
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called every time the Trainer logs data.
        """
        res_before = torch.cuda.memory_reserved(0)
        free_memory()
        res_after = torch.cuda.memory_reserved(0)
        freed = res_before - res_after
        print(f'Freed {freed}.')


def train_gpt2(config):
    """Trains a GPT-2 model.

    For the full set of parameters, see configs/config_train.yaml as well as
    https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments

    Args:
      config: A yaml config dict created by train_gpt2.parse_args()
    """
    set_seed(config['training_args']['seed'])
    pytorch_bs()
    device = get_device()

    # create tokenizer
    tokenizer = get_tokenizer()
    # create dataset(s)
    datasets = get_datasets(tokenizer=tokenizer, **config['custom_args'])
    train_only = config['custom_args']['val_size'] == 0
    if train_only:
        train_dataset = datasets
    else:
        train_dataset, val_dataset = datasets

    # create model
    checkpoint_dir = config['checkpoint']['checkpoint']
    model = get_model(checkpoint_dir, tokenizer).to(device)

    # training arguments
    if checkpoint_dir:
        config['training_args']['resume_from_checkpoint'] = checkpoint_dir

    training_args = transformers.TrainingArguments(**config['training_args'])

    # trainer
    # the padding token makes no difference
    data_collator = get_dynamic_padding_collator(tokenizer.pad_token_id)

    trainer_args = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'data_collator': data_collator,
        'callbacks': [Garbage_collector_callback],
    }
    if not train_only:
        trainer_args['eval_dataset'] = val_dataset

    trainer = transformers.Trainer(**trainer_args)
    if checkpoint_dir:
        trainer.train(checkpoint_dir)
    else:
        trainer.train()
