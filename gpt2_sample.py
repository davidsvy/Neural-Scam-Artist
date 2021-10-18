import argparse
from gpt2_.text_generation import generate_from_config
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', '-c', default='configs/gpt2_sample.yaml',
        help='Path to the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Check checkpoint_dir
    checkpoint_dir = config['other_args'].get('checkpoint_dir')
    if not checkpoint_dir:
        raise ValueError(
            'A valid checkpoint directory must be provided to generate text.')

    if not os.path.isdir(checkpoint_dir):
        raise ValueError(
            'The provided checkpoint directory does not exist.')

    # Check target_file
    target_file = config['other_args'].get('target_file')
    if not target_file:
        raise ValueError(
            'The path where the generated text will be stored was not given.')
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    if os.path.isfile(target_file):
        os.remove(target_file)

    return config


def main():
    config = parse_args()
    generate_from_config(config)


if __name__ == '__main__':
    main()
