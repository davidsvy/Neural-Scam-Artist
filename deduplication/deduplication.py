import csv
import datasketch
import nltk
from nltk.util import ngrams
import numpy as np
import os
import textstat
from tqdm import tqdm


class Deduplicator(object):
    """Used for detecting and removing duplicate documents.

    This is a very basic application of MinHash and LSH
    using the datasketch library. I have not experimented
    with hyperparameters such as different hash functions,
    Jaccard similarity threshlods etc, but anyone is
    welcome to. This implementation assumes that the entire
    dataset can fit into RAM, so it is unsuitable for
    extremely large datasets.

    Attributes:
      n_perm: (int) Number of hash functions used for MinHash and
        LSH. Large values of n_perm lead to more accurate results
        but to slower performance.
      n_ngram: (int) Size of ngrams that input textx are split into
        for MinHash and LSH. Default value is 5 as in
        https://arxiv.org/pdf/2107.06499.pdf
      tokenizer: (nltk.tokenize.regexp.RegexpTokenizer) Splits text
        into tokens and only takes into accounts letters (no digits
        or other characters).
      min_len: (int) Texts with number of words less than min_len are
        discarded.
    """

    __slots__ = ('n_perm', 'n_ngram', 'tokenizer', 'min_len')

    def __init__(self, n_perm=1024, n_ngram=5, min_len=10):
        """Initializes a Deduplicator module.

        Args
          n_perm: (int) Number of hash functions used for MinHash and
            LSH. Large values of n_perm lead to more accurate results
            but to slower performance.
          n_ngram: (int) Size of ngrams that the input texts are split into
           for MinHash and LSH. Default value is 5 as in
           https://arxiv.org/pdf/2107.06499.pdf
          min_len: (int) Texts with number of words less than min_len are
           discarded.
        """
        self.n_perm = n_perm
        self.n_ngram = n_ngram
        self.min_len = min_len
        self.tokenizer = nltk.tokenize.RegexpTokenizer('[a-z]{2,}')

    def read_file(self, source_file, n_texts=-1):
        """Reads the entire corpus from a csv file as a list of strings.

        Args:
          source_file: (str) Path to a csv file that contains the entire corpus.
          n_texts: (int) If provided, only the first n_texts samples will be
            read from the corpus.

        Returns:
          corpus: (list) List of samples from the dataset.
        """
        if not (isinstance(source_file, str) and source_file.endswith('.csv')):
            raise ValueError('source_file must be a path to a valid csv file')

        if not os.path.isfile(source_file):
            raise ValueError(f'{source_file} does not exist')

        corpus = []
        print(f'Reading dataset from {source_file}...')
        with open(source_file, 'r', encoding="utf-8") as file:
            reader = csv.DictReader(file, fieldnames=['url', 'text'])
            for idx, row in enumerate(reader):
                corpus.append(row['text'])
                if n_texts > 0 and idx > n_texts:
                    break

        return corpus

    def save_file(self, target_file, corpus):
        """Stores a list of strings inside a csv file.

        Args:
          target_file: (str) Path to a csv file where the corpus will be stored.
          corpus: (list) List of strings
        """
        if not (isinstance(target_file, str) and target_file.endswith('.csv')):
            raise ValueError('target_file must be a path to a valid csv file')

        with open(target_file, 'w', encoding="utf-8") as file:
            file.truncate(0)
            writer = csv.writer(file)
            print(f'Saving deduplicated dataset at {target_file}...')
            for text in corpus:
                writer.writerow([text])

    def get_ngrams(self, text):
        """Transforms a string into list of overlapping word ngrams.

        Example: for input 'Somebody once told me' and n=3, the output is:
            ['Somebody once told', 'once told me']

        Args:
          text: (str) Input text

        Returns:
          ngrams_: (list) List of overlapping word ngrams. If the number of
            words inside the string is less than min_len, None is returned.
        """
        tokens = self.tokenizer.tokenize(text.lower())
        # Discard input if it is not long enough
        if len(tokens) < self.min_len:
            return None
        ngrams_ = ngrams(tokens, self.n_ngram)
        ngrams_ = [' '.join(x).encode('utf8') for x in ngrams_]

        return ngrams_

    def hash_text(self, text):
        """Hashes input string with MinHash.

        Args:
          text: (str) Input text

        Returns:
          minhash: datasketch.MinHash object
        """
        ngrams_ = self.get_ngrams(text)
        if not ngrams_:
            return None
        minhash = datasketch.MinHash(num_perm=self.n_perm)
        # tiny improvement in speed compared to minhash.update
        minhash.update_batch(ngrams_)
        # The documentation claims that this saves memory usage
        minhash = datasketch.LeanMinHash(minhash)

        return minhash

    def lsh(self, corpus):
        """Constructs an LSH index from a text corpus.

        For the love of god, I could not make the computation parallel.
        However, it turns out that sequential computation is still
        sufficiently fast.

        Args:
          corpus: (list) List of strings. Each string is a sample from the
            dataset.

        Returns:
          lsh: datasketch.MinHashLSH object.
          signatures: (list) List of hash signatures for each sample in the
            corpus.
        """
        signatures = []
        lsh = datasketch.MinHashLSH(threshold=0.5, num_perm=self.n_perm)
        print('Constructing LSH Index...')
        for idx, text in enumerate(tqdm(corpus)):
            minhash = self.hash_text(text)
            signatures.append(minhash)
            if minhash:
                lsh.insert(idx, minhash)

        return lsh, signatures

    def construct_graph(self, lsh, signatures):
        """Constructs a graph of duplicates over the corpus.

        Each node corresponds to a single sample from the dataset,
        whereas an edge between nodes means that the respective samples
        are potential duplicates.

        Args:
          lsh: datasketch.MinHashLSH object.
          signatures: (list) List of hash signatures for each sample in the
            corpus.

        Returns:
          adjacency_list: (list) Adjacency list of the created graph.
          status: (np.array) Array with the same size as the input corpus.
            For a given index, -1 means that the signature of the respective
            sample could not be constructed.
        """
        status = np.zeros(len(signatures), dtype=np.int)
        adjacency_list = [[] for _ in range(len(signatures))]

        print('Creating duplicate graph...')
        for idx, signature in enumerate(tqdm(signatures)):
            if status[idx] != 0:
                continue

            if not signature:
                # mark samples than could not be hashed
                status[idx] = -1
                continue

            duplicates = lsh.query(signature)
            for duplicate in duplicates:
                # mark potential duplicates so that the for loop skips them
                status[duplicate] = 1
                adjacency_list[duplicate].append(idx)
            adjacency_list[idx] = duplicates

        return adjacency_list, status

    def get_connected_components(self, adjacency_list, status):
        """Locates connected components of a given graph using DFS.

        Args:
          adjacency_list: (list) Adjacency list of the given graph.
          status: (np.array) Array with the same size as the input corpus.
            For a given index, -1 means that the signature of the respective
            sample could not be constructed. These elements are ignored and
            not included in any connected component.

        Returns:
          connected_components: (list) Each element of the list is a connected
            component, ie a list of integer nodes.
        """
        n_nodes = len(adjacency_list)
        visited = [False] * n_nodes
        connected_components = []

        def recursive_visit(node):
            visited[node] = True
            connected_components[-1].append(node)
            for child in adjacency_list[node]:
                if not visited[child]:
                    recursive_visit(child)

        print('Detecting connected components...')
        # simple DFS implementation
        for node in range(n_nodes):
            if visited[node] or status[node] == -1:
                continue

            connected_components.append([])
            recursive_visit(node)

        return connected_components

    def select_component_node(self, component):
        """Selects single text from a list based on readability criterion.

        The goal of this function is to pick the most easily readable sample
        inside a set. The Gunning fog index was chosen arbitrarily
        (https://en.wikipedia.org/wiki/Gunning_fog_index).
        One is welcome to experiment with other functions.
        For more see https://pypi.org/project/textstat/


        Args:
        component: (list) A list of integer nodes belonging to the same
            connected component inside the duplication graph

        Returns:
        selected_text: (str) The selected text.
        """
        if len(component) == 1:
            return component[0]

        measures = []
        for text in component:
            measure = textstat.gunning_fog(text)
            measures.append(measure)

        selected_text = component[np.argmin(measures)]

        return selected_text

    def deduplicate_components(self, corpus, components):
        """Deduplicates corpus given its samples and connected components.

        Args:
          corpus: (list) List of strings. Each string is a sample from the
            dataset.
          components: (list) Each element of the list is a connected
            component of the duplication graph, ie a list of integer nodes.

        Returns:
          selected: (list) List of selected text samples, one for each
            connected component.
        """
        selected = []
        print('Selecting most suitable samples...')
        for idxs in components:
            component = [corpus[idx] for idx in idxs]
            selected.append(self.select_component_node(component))
        print(
            f'Selected {len(selected)} out of {len(corpus)} samples in total.')

        return selected

    def deduplicate(self, source_file, target_file, n_texts=-1):
        """Removes duplicates from a given text corpus.

        On Google Colab, Runtime is roughly 30 minutes for the entire corpus.

        Args:
          source_file: (str) Path to a csv file that contains the unprocessed 
            corpus.
          target_file: (str) Path to a csv file where the deduplicated corpus 
            will be stored.
          n_texts: (int) If provided, only the first n_texts samples will be
            read from the unprocessed corpus.
        """
        corpus = self.read_file(source_file, n_texts)
        lsh, signatures = self.lsh(corpus)
        adjacency_list, status = self.construct_graph(lsh, signatures)
        components = self.get_connected_components(adjacency_list, status)
        deduplicated_corpus = self.deduplicate_components(corpus, components)
        self.save_file(target_file, deduplicated_corpus)
