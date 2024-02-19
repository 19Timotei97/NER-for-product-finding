# Step 5
# I should use the model to predict some entities from the unseen pages
from typing import List, Dict
import ssl
import urllib.request
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from transformers import pipeline
from create_dataset import get_content_from_url
from nltk.tokenize import sent_tokenize
import pandas as pd
import json
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_links_file', type=str)
parser.add_argument('--path_to_save_json_result_file', type=str)


# Inspired from
# https://huggingface.co/docs/transformers/tasks/token_classification
class Predictor:
    def __init__(self):
        """Creates a very simple predictor, for testing the model

        """
        self.ssl_context = ssl.SSLContext()
        self.header = {'User-Agent': 'Mozilla/5.0'}
        self.ner = pipeline("ner", model="TimoteiB/DistilBERT_NER_furniture")

    def grab_url_links(self, url: str) -> List[str]:
        """Retrieves the links from a URL given as an argument.

        Args:
            url (str): The URL from which the links will be grabbed
        """
        # Grab the url of that first page
        urls_in_url = [url]

        req = urllib.request.Request(url, headers=self.header)
        html = urllib.request.urlopen(req, context=self.ssl_context).read()
        soup = BeautifulSoup(html, 'lxml')

        # Find all links / pages in a URL
        links_in_url = soup.find_all('a')  # get all a-href

        for link in links_in_url:
            # Create an absolute path to the link
            internal_url = urljoin(url, link.get('href'))

            # Add them to the list
            urls_in_url.append(internal_url)

        return urls_in_url

    def predict_ner(self, urls: List[str]) -> Dict:
        """The method used to run inference on the content of URLs

        Args:
            urls (List[str]): A list of URLs to be tested
        """
        # The complexity of this method is horrible, but it's just for testing the model
        entities_in_urls = {}

        for url in urls:
            logging.info(f"Got url {url}\n...")

            # Retrieve the list of urls from a main url
            url_list = self.grab_url_links(url)
            logging.info(f"Got {len(url_list)} urls from the main url")
            url_entities = []

            for _url in url_list:
                text = get_content_from_url(_url)
                # Grab sentences of that URL
                content_sentences = sent_tokenize(text)

                entities = []
                for sentence in content_sentences:
                    # Run the inference pipeline on the content
                    preds = self.ner(sentence)

                    logging.info(f"Got predictions: {preds}")
                    all_entities = []

                    for pred in preds:
                        if pred['entity'] == 'I-PRODUCT' or pred['entity'] == 'B-PRODUCT':
                            # If it's at least better than just guessing the products
                            if pred['score'] > 0.5:
                                all_entities.append(pred['word'])

                    entities.extend(all_entities)

                url_entities.extend(entities)

            entities_in_urls[url] = url_entities

        return entities_in_urls

    def predict_ner_page(self, text: str) -> List[str]:
        """Runs inference pipeline on a single page

        Args:
            text (str): The page on which inference will be run
        """
        entities = []

        # TODO: remove code repetition, use code from 'predict_ner'
        # TODO: check how to create a more elegant solution to this
        # Limit to the maximum length of the model
        if len(text) > 512:
            text = text[:512]
        preds = self.ner(text)

        # logging.info(f"Got predictions: {preds}")

        for pred in preds:
            if pred['entity'] == 'I-PRODUCT' or pred['entity'] == 'B-PRODUCT':
                # If it's at least better than just guessing the products
                if pred['score'] > 0.5:
                    entities.append(pred['word'])

        return entities


# Other ideas I had:
# Maybe create a dict -> json with url -> entities (for links that actually work)
# Another idea would be to use the model directly
"""
tokenizer = AutoTokenizer.from_pretrained(r'.\furniture_ner_project\model_distilbert_ner_finetuned')
model = AutoModelForTokenClassification.from_pretrained(r'.\furniture_ner_project\model_distilbert_ner_finetuned', num_labels=3)

with open(links_file, 'r') as file:
    links = file.readlines()
# df = pd.read_csv(links_file)
# links = df['max(page)'].tolist()

for url in links:
    content = get_content_from_url()
    # Split into sentences
    content_sentences = sent_tokenize(text)
    
    for sentence in content_sentences:
        tokens = tokenizer(sentence)
        torch.tensor(tokens['input_ids']).unsqueeze(0).size()

        predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), 
                            attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
        # Get the class with the maximum probability
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        predictions = [label_list[i] for i in preds]

# Decode the tokenizer results from input ids  to words
words = tokenizer.batch_decode(tokens['input_ids'])
# Optionally, save to a csv file
# pd.DataFrame({'ner': predictions, 'words': words}).to_csv('predictions.csv')
"""


def retrieve_entities():
    args = parser.parse_args()

    if args.path_to_links_file.endswith('.csv'):
        df = pd.read_csv(args.path_to_links_file)
        links = df['max(page)'].tolist()
    elif args.path_to_links_file.endswith('.txt'):
        with open(args.path_to_links_file, 'r') as file:
            links = file.readlines()
    else:
        logging.error("Got unrecognized file! Exiting")
        return

    logging.info(f"Got {len(links)} links from file.")
    predictor = Predictor()

    logging.info("Starting the prediction...")
    entities = predictor.predict_ner(links)

    # Save to json file
    # This is just a temporary 'feature', the json is way too big
    with open(args.path_to_save_json_result_file, 'w', encoding='utf-8') as json_file:
        json.dump(entities, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    logging.warning("THIS IS JUST A SIMPLE MODEL TESTING, I SUGGEST RUNNING visualize.py AND HEAD to http://127.0.0.1:5000!")
    retrieve_entities()
