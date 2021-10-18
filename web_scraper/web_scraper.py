import aiohttp
import asyncio
from bs4 import BeautifulSoup
import email
import nest_asyncio
import requests
import urllib

import csv
import os
from utils.reg import Regex_collection
from tqdm import tqdm


class Web_scraper(object):
    """A web scraper that downloads emails from https://antifraudintl.org/

    Atributes:
      main_page: (str) The main page of the scraped website.
      batch_size: (int) The maximum number of urls that can be
        accessed concurrently
      reg: (Regex_collection) Regexes for parsing urls and emails

    Typical usage example:
      web_scraper = Web_scraper()
      web_scraper.scrape() 
    """

    __slots__ = ('main_page', 'batch_size', 'reg')

    def __init__(self, batch_size=100):
        """Initializes a Web_scraper module.

        Args:
          batch_size: (int) The maximum number of urls that can be
            accessed concurrently
        """
        self.main_page = 'https://antifraudintl.org/'
        self.batch_size = batch_size
        self.reg = Regex_collection()
        nest_asyncio.apply()

    def set_batch_size(self, batch_size):
        """Updates the batch size to given value.

        Args:
          batch_size: (int) The updated batch size value
        """
        if not (isinstance(batch_size, int) and batch_size > 0):
            raise ValueError('The batch size must be a positive integer')
        self.batch_size = batch_size

    def head_from_email(self, text):
        """Extracts email body and removes metadata.

        Args:
          text: (str) Unprocessd email text

        Returns:
          Processed email string
        """
        text_ = email.message_from_string(text)

        # Use the email lib if it functions correctly
        # Else, manually remove metadata
        if not text_.defects and len(text_._headers) > 3:
            head = text_._payload
        else:
            head = text.split('\n')
            offset = 0
            for idx, line in enumerate(head):
                if self.reg.prefix_re.search(line):
                    offset = idx + 1

            head = '\n'.join(head[offset:])  # .strip()

        # Remove whitespace at the end of the text
        match = self.reg.suffix_re.search(head)
        if match:
            head = head[:match.span()[0]]

        return head

    def find_all_topics(self):
        """Returns a list of scam category webpages.

        Each element of the list is a dictionary that contains a url of the 
        topic and its title.
        """
        page = requests.get(self.main_page)
        soup = BeautifulSoup(page.content, "html.parser")
        topic_block = soup.find_all(
            'div', class_='block block--category block--category19')[0]
        all_links = topic_block.find_all('a', href=True)

        topics = [a for a in all_links if self.reg.is_topic(a)]
        topics = [
            {'url': urllib.parse.urljoin(self.main_page, a['href']), 'name': a.contents[0]} for a in topics
        ]

        return topics

    async def page_to_threads(self, url, session):
        """Collects all thread urls inisde a page

        Args:
          url: (string) A valid url that contains multiple threads 
            (eg https://antifraudintl.org/forums/next-of-kin.23/page-69)
          session: (aiohttp.ClientSession) Necessary for parallel execution.

        Returns: 
          (List) A url for each thread that belongs to arg "url".
        """
        try:
            async with session.get(url=url) as response:
                page = await response.read()
                soup = BeautifulSoup(page.decode('utf-8'), 'html5lib')

                threads = [a['href'] for a in soup.find_all(
                    'a', href=True) if self.reg.is_thread(a)]
                threads = set(threads)
                # concat string to create absolute address
                threads = [urllib.parse.urljoin(url, t) for t in threads]
                return threads

        except Exception as e:
            print(e)
            return None

    async def topic_to_threads(self, url, size=-1):
        """Collects all threads inisde a topic page

        Args:
          url: (string) A valid url that contains multiple threads 
            (eg https://antifraudintl.org/forums/next-of-kin.23/)
          size: (int) Maximum number of pages to examine inside the topic.

        Returns: 
          (list) A list of thread urls.
        """
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")

        links = [a['href'] for a in soup.find_all('a', href=True)]
        # Find the number of pages belonging to the given topic
        nums = [self.reg.num_re.search(l) for l in links]
        nums = [int(n.group(1)) for n in nums if n]
        n_pages = max(nums) if nums else 0

        # Create urls for all the numbered pages
        children = [url]
        children += [urllib.parse.urljoin(url, f'page-{idx}')
                     for idx in range(2, n_pages + 1)]
        if size > 0:
            children = children[:size]

        res = []
        # split the numbered pages into batches
        # and collects urls for each batch concurrently
        for idx in range(0, len(children), self.batch_size):
            async with aiohttp.ClientSession() as session:
                res_batch = await asyncio.gather(
                    *[self.page_to_threads(
                        child, session) for child in children[idx: idx + self.batch_size]]
                )
            # print(res_batch)
            # flatten the batcehs into a single list
            for sub in res_batch:
                res += sub

        return res

    def find_all_threads(self, n_topics=-1, target_file=None):
        """Collects all thread urls in the entire forum and saves them inside a csv file.

        Each url is stoted in a seperate line.

        Args:
          n_topics: (int) If positive, only the n_topics first forum topics will 
            be scraped.
          target_file: (str) Path to the csv file where the urls will be stored.

        Returns:
          (str) Path to the saved csv file.
        """
        if not target_file:
            target_file = 'saved_urls.csv'
        elif not (isinstance(target_file, str) and target_file.endswith('.csv')):
            raise ValueError('target_file must be a csv file')

        label_file = target_file[:-4] + '_label.csv'

        print('Collecting all topics...')
        topics = self.find_all_topics()
        if n_topics > 0:
            topics = topics[:n_topics]
        line_cnt = [0]

        print('Collecting all thread URLs...')
        with tqdm(total=len(topics)) as bar, open(target_file, 'w', encoding='utf-8') as f:
            f.truncate(0)
            writer = csv.writer(f, delimiter='\n')

            for topic in topics:
                bar.set_description(f"Collecting {topic['name']}")
                # receive a list of url files for a specific topic
                topic_threads = asyncio.run(
                    self.topic_to_threads(topic['url'], size=-1))
                topic_threads = list(filter(None, topic_threads))
                # save the urls inside the csv file
                writer.writerows([topic_threads])
                line_cnt.append(line_cnt[-1] + len(topic_threads))
                bar.update(1)

        labels_data = [f"{cnt} {label['name']}" for cnt,
                       label in zip(line_cnt, topics)]
        # create metadata label file where each line line has the structure:
        # line where topic urls start -> space -> name of the topic
        with open(label_file, 'w', encoding='utf-8') as f:
            f.truncate(0)
            writer = csv.writer(f, delimiter='\n')
            writer.writerows([labels_data])

        print(f'Collected {line_cnt[-1]} URLs in total')
        print(f'URLs saved at {target_file}')
        print(f'URL labels saved at {label_file}')

        return target_file

    async def url_to_text(self, url, session):
        """Collects emails from a thread page.

        This is a very basic implementation that locates all messages inside
        the thread, finds the first message that contains email metadata ( 
        Return-Path:, Date: etc) and returns the said message. The rest of the
        messages are either personal comments or duplicates of the found email.

        Args:
          url: (string) A valid thread url 
            (eg https://antifraudintl.org/threads/zuman-balanji.84835/)
          session: (aiohttp.ClientSession) Necessary for parallel execution.

        Returns:
          The string of the found email. If no email is found or some error
            occurs None is returned
        """
        try:
            async with session.get(url=url) as response:
                page = await response.read()
                soup = BeautifulSoup(page.decode('utf-8'), 'html5lib')
                # remove Javascript from the webpage for easier parsing
                for script in soup(["script", "style", "blockquote"]):
                    script.decompose()
                msgs = soup.find_all('article')
                msgs = [msg.text for msg in msgs]

                email_, match = None, None
                for msg in msgs:
                    match = self.reg.email_re.search(msg)
                    if not match:
                        continue
                    email_ = msg[match.span()[0]:].strip()
                    email_ = self.head_from_email(email_)
                    break

                return email_

        except Exception as e:
            print(e)
            return None

    async def csv_to_text(self, source_file, target_file, max_urls=-1):
        """Reads urls from source_file and returns emails that they contain.

        Args:
          source_file: (str) Path to a valid csv file that contains thread urls
            seperated by newline. Such a file can be created by find_all_threads.
          target_file: (str) Path tovalid csv file where the emails will be 
            stored.

        Returns:
          target_file: (str) Path to the csv file where the dataset will be
            stored.
          failed_urls: (list) List of string urls for which url_to_text failed
            to retrieve an email.
        """
        # number of urls inside source_file
        n_urls = sum(1 for _ in open(source_file, 'rb'))
        if max_urls > 0:
            n_urls = min(n_urls, max_urls)
        n_success = 0
        n_total = 0
        failed_urls = []

        print(f'Collecting text from thread urls...')
        with tqdm(total=n_urls) as bar:
            with open(source_file, 'r', encoding='utf-8') as fs, open(target_file, 'w', encoding='utf-8') as ft:
                ft.truncate(0)
                reader = csv.reader(fs, delimiter='\n')
                # , delimiter='\n')
                writer = csv.DictWriter(ft, ['url', 'text'])
                input_batch = []
                output_batch = []

                for step, url in enumerate(reader, 1):

                    if url:
                        input_batch.append(url[0])
                    else:
                        continue

                    if not (step % self.batch_size == 0 or step == n_urls):
                        continue

                    async with aiohttp.ClientSession() as session:
                        text_batch = await asyncio.gather(
                            *[self.url_to_text(url, session) for url in input_batch]
                        )

                    for u, t in zip(input_batch, text_batch):
                        if t:
                            row = {'url': u, 'text': t}
                            output_batch.append(row)
                        else:
                            failed_urls.append(u)

                    n_total += len(input_batch)
                    n_success += len(output_batch)
                    writer.writerows(output_batch)
                    bar.update(len(input_batch))
                    input_batch = []
                    output_batch = []
                    if n_total >= n_urls:
                        break

        print(
            f'Successfully collected {n_success} emails from {n_urls} threads!!!')
        print(f'The dataset was saved at {target_file}')
        return target_file, failed_urls

    def scrape(self, from_csv=False, source_file=None, target_file=None,
               n_topics=-1, n_urls=-1):
        """Saves scraped emails inside a target_file.

        On Google Colab, runtime is roughly 2 hours for the entire corpus.

        Args:
          from_csv: (Bool) If True, urls are read from source_file and then 
            scraped for emails. Otherwise, thread urls are first collected and
            stored inside source_file and only then scraped for emails.
          source_file: Path to a csv file with thread urls. Such a file can be
            created by find_all_threads. If from_csv is True, then only urls
            from source_file are scraped. Otherwise, urls are first collected
            and then stored inside source_file. If no path is provided, the 
            default path is saved_urls.csv.
          target_file: Path to a csv file where the final text dataset will be
            stored.
          n_topics: If provided, only the first n_topics topics will be searched
            for thread urls.
          n_urls: If provided, only the first n_urls urls will be searched
            for emails.

        Returns:
          failed_urls: (list) All urls for which emails could not be retrieved.
        """
        if target_file:
            if not (isinstance(target_file, str) and target_file.endswith('.csv')):
                raise ValueError(
                    'target_file must be a path to a valid csv file')
        else:
            target_file = 'dataset.csv'

        if source_file:
            if not (isinstance(source_file, str) and source_file.endswith('.csv')):
                raise ValueError(
                    'source_file must be a path to a valid csv file')
        else:
            source_file = 'saved_urls.csv'

        if from_csv:
            if not os.path.isfile(source_file):
                raise ValueError(f'{source_file} does not exist')

            # scrape text from all urls inside source_file
            failed_urls = asyncio.run(self.csv_to_text(
                source_file, target_file, n_urls))

        else:
            # collect all thread urls and store them inside source_file
            source_file = self.find_all_threads(
                n_topics=n_topics, target_file=source_file)
            # scrape text from all urls inside source_file
            failed_urls = asyncio.run(self.csv_to_text(
                source_file, target_file, n_urls))

        return failed_urls
