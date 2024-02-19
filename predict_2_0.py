import csv
import json
import urllib3
import sys
import logging

from inference import NERproductFinder
from crawler import Crawler

# Stops warnings related to invalid SSL certificates
urllib3.disable_warnings()

# Create the logger for displaying information
# logger = logging.getLogger('furniture-trf-logger')
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.INFO)

# Most of the time, the headers seem to contain product names / titles
# For example:
# <body><h1>ProductPageName</h1><h2>Great Lithuanian Pillow Crest</h2></body>
INTERESTING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5']
CLS_THRESHOLD = 0.5  # The default one, could be changed
KEEP_THRESHOLD = 0.6

prod_finder = NERproductFinder(
    device='cpu',
    checkpoint='checkpoints/0.6805472219712101.dat',  # Save your very own checkpoint and use it here
    backbone='distilbert-base-uncased',
    tokenizer='distilbert-base-uncased',
)

crawler = Crawler(
    html_tags=INTERESTING_TAGS,
    # timeout=10  # A timeout variable could be introduced to only test a URL for a limited nb of seconds
)

results_dict = {}

# Parse the CSV file of URLs
with open('data/furniture stores pages.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\n')

    # Skip the CSV's header
    next(csvreader)

    for row in csvreader:
        # Crawl the URL
        crawler_results = crawler.crawl(row[0], max_depth=0, max_urls=50)

        if not row[0] in results_dict:
            results_dict[row[0]] = []

        # For each crawled URL
        for crawler_result in crawler_results:
            found_product = False

            # Check content of headings from most to least important (h1->h5)
            for tag in INTERESTING_TAGS:
                # Found possible product name in heading -> stop
                if found_product:
                    break

                # If tag is found within the site's source code
                if tag in crawler_result['tags']:
                    # Look at the values from all tag's occurrences
                    for text in crawler_result['tags'][tag]:

                        # Run inference on the tag's value
                        inf_result = prod_finder.run_inference(text, threshold=CLS_THRESHOLD)
                        is_product = inf_result['output']
                        confidence = inf_result['confidence']

                        # Only log results that exceed the confidence threshold (at least better than randomly guessing)
                        if is_product and confidence > KEEP_THRESHOLD:

                            # Check so that we don't add the URL's result multiple times
                            if crawler_result['url'] not in results_dict:
                                results_dict[crawler_result['url']] = []

                            results_dict[crawler_result['url']].append({'product': text,
                                                                        'is_product?': is_product,
                                                                        'confidence': confidence})

                            # We found the product name - so we don't need to go any further
                            found_product = True

        with open("outputs/products_found.json", 'w') as f:
            json.dump(results_dict, f)
