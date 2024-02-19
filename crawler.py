import requests
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urljoin
import logging
import re
import ssl
from typing import Optional, List


class Crawler:
    def __init__(self, html_tags: List[str]):
        """Creates the web crawler

        Args:
            html_tags (List[str]): list of html tags that may contain useful information
        """
        self.html_tags = list(set(html_tags + ['a']))
        self.ssl_context = ssl.SSLContext()
        self.header = {'User-Agent': 'Mozilla/5.0'}

        # Testing
        # self.header = {'User-Agent': 'Mozilla/5.0 (Macintosh; '
        #                              'Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
        #                              'Chrome/104.0.5112.79 Safari/537.36'}
        self.hostnames = ""

    def get_hostname(self, addr: str) -> Optional[str]:
        """Retrieves the hostname and tld from a given URL (relative or absolute)

        Args:
            addr (str): target address (relative or absolute)

        Returns:
            Optional[str]: the hostname, if found - otherwise None
        """
        """
        Here a regular expression is used to find strings like example.com or sub-domain.example123.com.
        How does it work?
        
        There's 2 parts:
            1) (?:[a-zA-Z0-9\\-]+\\.)+ -> one or more occurrences (+ at the end of the [] group) of alphanumeric chars 
                                            or hyphens(-) followed by a dot. The ?: basically means that the expression
                                            is treated as a whole, rather than being a capturing group, aka a part of
                                            the expression used for other matches.
            2) w+ -> matches one or more word chars 
        """
        self.hostnames = re.findall('((?:[a-zA-Z0-9\\-]+\\.)+\\w+)', addr)

        return self.hostnames[0] if len(self.hostnames) != 0 else None

    def get_absolute_from_relative_url(self, rel_addr: str, parent_scheme: str, parent_hostname: str, url: Optional[str]) -> str:
        """Converts a relative URL to its absolute counterpart

        Args:
            rel_addr (str): the relative address
            parent_scheme (str): parent scheme (https or http or other)
            parent_hostname (str): parent hostname
            url (Optional[str]): just for testing purposes, trying the whole url

        Returns:
            str: the absolute URL
        """
        # ensure '/about' becomes 'about'
        while len(rel_addr) > 0 and rel_addr[0] == '/':
            rel_addr = rel_addr[1:]

        # map 'about' to 'https://sth.tld/about'
        abs_addr = f'{parent_scheme}://{parent_hostname}/{rel_addr}'
        logging.info(f"Converted [{rel_addr}] -> [{abs_addr}]")
        print(f"Converted [{rel_addr}] -> [{abs_addr}]")

        abs_addr_test = urljoin(url, rel_addr.get('href'))
        logging.info(f"Converted [{rel_addr}] -> [{abs_addr_test}]")
        print(f"Converted [{rel_addr}] -> [{abs_addr_test}]")

        return abs_addr

    def crawl(self, url: str, max_depth: int = 0, max_urls: Optional[int] = None) -> List[dict]:
        """Crawls the targeted URL and continues by discovering links in the given page

        Args:
            url (str): the URL for crawling
            max_depth (int, optional): the maximum depth for BFS crawling. Defaults to 0.
            max_urls (Optional[int], optional): a maximum number of URLS to discover and crawl automatically.
                                                Defaults to None.

        Returns:
            List[dict]: a list of dictionaries which contain per-URL crawl information (tags, values, etc.)
        """
        num_urls = 0

        urls_queue = set()
        urls_queue.add((url, max_depth))
        visited = {url: True}

        # Retrieve the hostname
        parent_hostname = self.get_hostname(url)
        parent_scheme = 'https' if 'https' in url else 'http'

        results = []

        # Start crawling while there's still urls in the queue
        while len(urls_queue) > 0:
            (curr_url, curr_depth) = urls_queue.pop()

            # Get the info from the url
            curr_result = self.crawl_page(url=curr_url)

            # Check if we can still go in depth
            num_urls += 1
            if max_urls is not None and num_urls >= max_urls:
                break

            # Check if we have other links
            if curr_depth > 0 and 'a' in curr_result['tags']:

                # Start going through each link
                for a_link in curr_result['tags']['a']:

                    curr_a_link = a_link
                    crt_a_link_hostname = self.get_hostname(curr_a_link)

                    # Convert to absolute url format
                    if crt_a_link_hostname is None:
                        curr_a_link = self.get_absolute_from_relative_url(
                            curr_a_link, parent_scheme, parent_hostname, a_link)

                    # Mark the link as being visited if not already
                    if curr_a_link not in visited:
                        if self.get_hostname(curr_a_link) == parent_hostname:
                            # Add the link with its depth
                            urls_queue.add((curr_a_link, curr_depth - 1))
                            visited[curr_a_link] = True

            results.append(curr_result)

        return results

    def crawl_page(self, url: str) -> dict:
        """Crawls an individual page for tags of interest and generates a crawl report

        Args:
            url (str): the target URL address

        Returns:
            dict: a crawl report which includes tags, values, etc.
        """

        result = {'url': url, 'status': None, 'tags': {}}

        logging.info(f'Crawling URL: [{url}]')

        # queries the target URL
        response = None
        try:
            # 2 ways of accessing an url:
            # 1. Using requests
            response = requests.get(url, verify=False, headers=self.header, timeout=10)

            # 2. Using urllib's request
            req = urllib.request.Request(url, headers=self.header)
            html = urllib.request.urlopen(req, context=self.ssl_context).read()
        except Exception as request_exception:
            logging.warning(
                f'Failed to crawl [{url}]: {str(request_exception)}')
            return result

        # Either way works
        # soup = BeautifulSoup(response.text, "html.parser")
        soup = BeautifulSoup(html, 'html.parser')

        # Locates HTML tags of interest within the page and extracts values
        possible_tags = soup.find_all(self.html_tags)

        # Go through each tag
        for possible_tag in possible_tags:
            tag_name = possible_tag.name
            tag_text = possible_tag.text.strip()

            # Get the link if it's a 'a' tag
            if possible_tag.name == 'a':
                href_location = possible_tag.get('href')

                if href_location is None:
                    continue
                tag_text = href_location

            # Check if we didn't check the tag already
            if tag_name not in result['tags']:
                result['tags'][tag_name] = []

            # Create the result object
            result['status'] = response.status_code
            # result['status'] = req.status_code
            result['tags'][tag_name].append(tag_text)

        return result
