# Step 2
# I should start the actual crawling from the good links I've gathered
from bs4 import BeautifulSoup
import urllib.request
import ssl
import pandas as pd
import re
from urllib.parse import urljoin
import trafilatura
import argparse
import logging

# Just asked ChatGPT what are the most recognized furniture types - certainly this is not the full list
furniture_products = ['sofa', 'couch', 'settee', 'chair', 'table', 'bed',
                        'desk', 'dresser', 'armchair', 'recliner', 'accent chair',
                        'table', '', 'side table', 'bedframe', 'headboard',
                        'mattress', 'cupboard', 'wardrobe', 'bookshelf',
                        'ottoman', 'coffee table', 'nightstand', 'dining table',
                        'sideboard', 'cabinet', 'entertainment center', 'futon',
                        'bench', 'stool', 'chaise lounge', 'loveseat']


parser = argparse.ArgumentParser()
parser.add_argument('--path_to_links_file', type=str, default='good_links.txt')
parser.add_argument('--path_to_save_csv', type=str, default='dataset.csv')


class DatasetCreator:
    def __init__(self, urls):
        self.urls = urls
        self.ssl_context = ssl.SSLContext()
        self.header = {'User-Agent': 'Mozilla/5.0'}

    def get_all_linked_furnitures_websites(self, url: str) -> list:
        # Start with the main url in the list.
        linked_furniture_urls = [url]
        logging.info(f"Trying url {url}...")

        try:
            req = urllib.request.Request(url, headers=self.header)
            html = urllib.request.urlopen(req, context=self.ssl_context).read()
            soup = BeautifulSoup(html, 'lxml')

            links_in_url = soup.find_all('a')  # get all a-href

            for link in links_in_url:
                # Similar to what join in Python does, it combines a main url with the hyperlink -> absolute link
                link = urljoin(url, link.get('href'))

                # If one of the furniture types is found in the link, add the link -> it may be useful
                if any([furniture_type in link for furniture_type in furniture_products]):
                    linked_furniture_urls.append(link)

            logging.info(f"Grabbed {len(linked_furniture_urls[1:])} useful links.")

        except Exception as e:
            logging.error(f"Error {str(e)} for url {url}")

        # Get 5 of all included links (or all if less than 5)
        if len(linked_furniture_urls) < 6:
            # 6 because it includes the main url
            logging.info("Less than 6 sites")

            return linked_furniture_urls

        return linked_furniture_urls[:6]

    def get_urls(self):
        if 0 == len(self.urls):
            logging.error("Please upload a valid list of urls (At least one!)")
            return

        # List of lists of urls
        url_with_urls = [self.get_all_linked_furnitures_websites(url) for url in self.urls]
        # Only leave the urls with at least a useful link (some of them don't have any of the furniture types)
        url_with_urls = [url for url_list in url_with_urls for url in url_list if len(url_list) > 1]

        return url_with_urls


# Tried many ways using bs4, which is cool, but rather difficult to tune to fit best
# Found this amazing thingy, trafilatura which does exactly what I want
# https://trafilatura.readthedocs.io/en/latest/quickstart.html
def get_content_from_url(url) -> str:
    html = trafilatura.fetch_url(url)
    url_content = re.sub(r" +", " ", str(trafilatura.extract(html, include_comments=False)))

    # There may be more elegant solutions for this, but I just got the text up until the
    #  max length of the tokenizer that will be used
    return url_content[:512]


def label_dataset():
    args = parser.parse_args()

    # Gather the links
    # with open('good_links.txt', 'r') as f:
    #     links = f.readlines()
    with open(args.path_to_links_file, 'r') as f:
        links = f.readlines()

    dataset_creator = DatasetCreator(links)
    url_with_urls = dataset_creator.get_urls()

    # Get and store the texts from the urls
    # Only limit to about 300 urls for convenience -> avoid a gigantic csv file
    texts = [{"url": url, "text": get_content_from_url(url)} for url in url_with_urls[:300]]

    df = pd.DataFrame(data=texts)

    # Remove urls with no content retrieved
    indices = df[df['text'] == 'None'].index
    df.drop(indices, inplace=True)
    # print(df[df.columns[0]].count())

    df.to_csv(path_or_buf=args.path_to_save_csv,
              sep=',',
              index=False)


if __name__ == '__main__':
    label_dataset()


# Stuff I tried and didn't work

# Attempt 1
# Tried using bs4 to grab all text in different headers and paragraphs
# Resulted in many text being wrongly formatted and lack of useful information
"""
# Function to extract text from a given URL
def extract_text_from_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, context=ssl.SSLContext()).read()

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Extract text content from relevant HTML elements
        # Adjust the HTML tags and classes based on the structure of the web page
        extracted_text = ""

        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # check to see if any furniture product specific words are found
            words = element.get_text().lower().split()
            if len(set(words).intersection(furniture_products)) > 0:
                extracted_text += element.get_text() + '\n'
                print("Found furniture text: ")
                print(extracted_text)

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in extracted_text.splitlines())

        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # drop blank lines
        extracted_text = '\n'.join(chunk for chunk in chunks if chunk)

        extracted_text = re.sub(r" +", " ", extracted_text)

        print("Success! Got text:")
        print(extracted_text)

        return extracted_text
    except Exception as e:
        print(f"Failed to retrieve content from {url}")
        print(f"An error occurred: {e}")

        return ""


texts = [{"url": link, "text": extract_text_from_url(link)} for link in links]

df = pd.DataFrame(data=texts)
df.to_csv(path_or_buf="./dataset_tuesday_1.csv",
          sep=',',
          index=False)
"""


# Attempt 2
# Tried various methods of combining bs4 to grab the text available
#  and then format it to get rid of extra spaces and '\n's.
# Again, the texts were poorly formatted and no useful information.
# I was missing the rule of actually looking for furniture types in the titles!
"""
texts = []
texts_1 = []
texts_2 = []


# ################################# DATASET CREATION VARIANTS ##############################
# Version 1
def get_text_from_url(url):
    try:
        # Doing it simply this way gave me an 403 code -> Forbidden
        # html = urllib.request.urlopen(url, context=ssl.SSLContext()).read()

        # It worked this way, by specifying the agent used to access the url
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, context=ssl.SSLContext()).read()

        soup = BeautifulSoup(html, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text.encode('utf-8')[:100]
    except Exception as e:
        print(f"Error for {url}")
        print(f"An error occurred: {e}")
"""

# ##################################################################################################

# Attempt 3
# Tried to get rid of elements that would not yield useful information.
# The information grabbed was again not filtered, so poorly formatted + lack of usefulness.
"""
# Version 2
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, context=ssl.SSLContext()).read()

        soup = BeautifulSoup(html, 'html.parser')

        texts = soup.findAll(string=True)

        visible_texts = filter(tag_visible, texts)

        return u" ".join(t.strip() for t in visible_texts)[:100]
    except Exception as e:
        print(f"Error for {url}")
        print(f"An error occurred: {e}")
"""


# ##################################################################################################

# Attempt 4
# I tried simply grabbing the texts from the paragraphs.
# Missing a lot of information from headers and lack of filtering.
"""
# Version 3
def extract_text_from_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, context=ssl.SSLContext()).read()

        soup = BeautifulSoup(html, 'html.parser')

        # Extract text from the webpage
        text = ' '.join([p.get_text() for p in soup.find_all('p')])

        return text[:100]
    except Exception as e:
        print(f"Error for {url}")
        print(f"An error occurred: {e}")


for link in links:
    # https://www.imfurniturestore.com/products/sofa - apparently stopped working ?
    print(link)
    texts.append({"url": link, "text": get_text_from_url(link)})
    print("Added to text_0")

    texts_1.append({"url": link, "text": text_from_html(link)})
    print("Added to text_1")

    texts_2.append({"url": link, "text": extract_text_from_url(link)})
    print("Added to text_2")

print(len(texts))
print(len(texts_1))
print(len(texts_2))

df = pd.DataFrame(data=texts)
df.to_csv(path_or_buf="./dataset.csv",
          sep=',',
          index=False)

df = pd.DataFrame(data=texts_1)
df.to_csv(path_or_buf="./dataset_1.csv",
          sep=',',
          index=False)

df = pd.DataFrame(data=texts_2)
df.to_csv(path_or_buf="./dataset_2.csv",
          sep=',',
          index=False)
"""