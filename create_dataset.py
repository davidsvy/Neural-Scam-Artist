import argparse
import os
from web_scraper.web_scraper import Web_scraper
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', '-c', default='configs/create_dataset.yaml',
        help='Path to the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    source_file = config['scraping_params']['source_file']
    target_file = config['scraping_params']['target_file']

    if not (isinstance(target_file, str) and target_file.endswith('.csv')):
        raise ValueError('target_file must be a path to a valid csv file')

    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not (isinstance(source_file, str) and source_file.endswith('.csv')):
        raise ValueError('source_file must be a path to a valid csv file')

    source_dir = os.path.dirname(source_file)
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)

    from_csv = config['scraping_params']['from_csv']
    if from_csv and not os.path.isfile(source_file):
        raise ValueError(f'{source_file} does not exist')

    return config


def main():
    config = parse_args()
    web_scraper = Web_scraper(**config['model_params'])
    web_scraper.scrape(**config['scraping_params'])


if __name__ == '__main__':
    main()
