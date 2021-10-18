import argparse
import os
from deduplication.deduplication import Deduplicator
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', '-c', default='configs/deduplicate_dataset.yaml',
        help='Path to the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    source_file = config['deduplication_params'].get('source_file')
    target_file = config['deduplication_params'].get('target_file')

    # Check source_file
    if not source_file:
        raise ValueError('Path to unprocessed corpus not provided')

    if not (isinstance(source_file, str) and source_file.endswith('.csv')):
        raise ValueError('source_file must be a path to a valid csv file')

    if not os.path.isfile(source_file):
        raise ValueError(f'{source_file} does not exist')

    # Check target_file
    if not (isinstance(target_file, str) and target_file.endswith('.csv')):
        raise ValueError('target_file must be a path to a valid csv file')

    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    return config


def main():
    config = parse_args()
    deduplicator = Deduplicator(**config['model_params'])
    deduplicator.deduplicate(**config['deduplication_params'])


if __name__ == '__main__':
    main()
