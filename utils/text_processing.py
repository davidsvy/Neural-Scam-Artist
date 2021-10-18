import csv
import pprint
import re


def clean_whitespace(str_):
    """Normalizes whitespace.

    Replaces consecutive newlines with single newline and all 
      other whitespace with a single space.

    Args:
      str_: (str) Input document.

    Returns:
      str_: (str) Output document with normalized whitespace.  
    """
    def sub_func(match):
        if match.group(0) == '\n':
            return '\n'
        return ' '

    str_ = str_.strip().lower() if str_.isupper() else str_.strip()
    str_ = '\n'.join(filter(None, map(str.strip, str_.split('\n'))))
    str_ = re.sub(r'\s+', sub_func, str_)

    return str_


def read_csv_file(source_file, n_samples=-1):
    """Reads a csv dataset.

    Args:
      source_file: (str) Path to a csv file.
      n_samples: (int) If given, only the first n_samples rows will 
        be read.

    Returns:
      corpus: (list) List of strings.
    """
    corpus = []
    with open(source_file, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        for step, row in enumerate(reader, 1):
            text = clean_whitespace(row[0])
            corpus.append(text)
            if n_samples > 0 and step >= n_samples:
                break

    return corpus


def dict_to_string(dict_):
    """Converts a dictionary into a pretty string    
    """
    return pprint.pformat(dict_)
