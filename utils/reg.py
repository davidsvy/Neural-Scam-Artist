import re


class Regex_collection(object):
    """Auxilliary class that stores regexes for parsing urls and emails
    """

    __slots__ = ('email_re', 'prefix_re', 'suffix_re', 'num_re', 'topic_re')

    def __init__(self):
        """Initializes a Regex_collection module
        """
        prefixes = [
            r'^Received',
            r'^Return(\ |-)Path',
            r'^Reply(\ |-)To',
            r'^Message(\ |-)ID',
            r'^Date',
            r'^From',
            r'^To',
            r'^Sent',
            r'^X(\ |-)SID(\ |-)PRA',
            r'^mailed(\ |-)by',
            r'^signed(\ |-)by',
            r'^Subject:',
        ]

        email_keywords = [
            r'\nReceived(\ |-)from',
            r'\nReturn(\ |-)Path',
            r'\nReply(\ |-)To',
            r'\nMessage(\ |-)ID',
            r'\nmailed(\ |-)by',
            r'\nsigned(\ |-)by',
            r'\nFrom:',
            r'\nTo:',
            r'\nSubject:',
            r'\nX(\ |-)SID(\ |-)PRA',
        ]

        email_re = f"({'|'.join(email_keywords)})"
        prefix_re = f"({'|'.join(prefixes)})"

        # Used for detecting emails amongst thread messages
        self.email_re = re.compile(email_re, flags=re.I)
        # Used for removing unnecessary email metadata
        self.prefix_re = re.compile(prefix_re, flags=re.I)
        # Used for removing unnecessary whitespace at the end of an email
        suffix_re = r'(\s{3,}Reply\s*|\s+Last edited)'
        self.suffix_re = re.compile(suffix_re, flags=re.I)
        #self.suffix_re = re.compile(r'\s{3,}Reply\s*', flags=re.I)
        # Used for detecting numbered thread pages
        # eg https://antifraudintl.org/forums/atm-card-scams.612/page-3
        self.num_re = re.compile(r'^\/forums\/.*page-(\d+)$')
        # Used for detecting topic pages
        # eg https://antifraudintl.org/forums/adoption-scams.710/
        self.topic_re = re.compile(r'\/forums\/.*\.\d+\/$')

    def is_thread(self, soup):
        """Returns True if a is a valid thread page
        eg https://antifraudintl.org/threads/zuman-balanji.84835/

        Args:
          soup: (BeautifulSoup object) A webpage
        """
        url = soup['href']
        flag = url.startswith('/threads/') and url.endswith('/')
        return flag

    def is_topic(self, soup):
        """Returns True if a is a valid topic page
        eg https://antifraudintl.org/forums/adoption-scams.710/

        Args:
          soup: (BeautifulSoup object) A webpage
        """
        flag = self.topic_re.search(soup['href'])
        flag = flag and len(soup.contents[0]) > 1
        return flag
