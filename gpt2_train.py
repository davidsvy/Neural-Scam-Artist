import argparse
from gpt2_.utils import train_gpt2
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', '-c', default='configs/gpt2_train.yaml',
        help='Path to the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    source_file = config['custom_args'].get('source_file')
    # Check source_file
    if not source_file:
        raise ValueError('Path to dataset not provided')

    if not (isinstance(source_file, str) and source_file.endswith('.csv')):
        raise ValueError('source_file must be a path to a valid csv file')

    if not os.path.isfile(source_file):
        raise ValueError(f'{source_file} does not exist')

    return config


def main():
    config = parse_args()
    train_gpt2(config)


if __name__ == '__main__':
    main()
