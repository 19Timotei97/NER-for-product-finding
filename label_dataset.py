# Step 3
# I should tag / label some of the products from the dataset

# ATTEMPT 5: kinda successfully?
# Use the words from the contents to find words from the titles of the urls grabbed
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as request
import ssl
import re
from nltk.tokenize import sent_tokenize
# import nltk # import it for the model download only!
# nltk.download('punkt')  # needs to be run at first run!
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--path_to_csv', type=str, default='dataset.csv')
parser.add_argument('--path_to_save_labeled_csv', type=str, default='labeled_dataset.csv')


class DatasetLabeller:
    def __init__(self, csv):
        self.csv = csv
        self.ssl = ssl.SSLContext()

    # This method could very well be outside the class, but wanted to keep the code kinda organized
    def grab_url_title(self, url):
        req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = request.urlopen(req, context=self.ssl).read()

        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title').string

            # Drop multiple whitespaces (+: 1 or more)
            title = re.sub(r" +", " ", title)

            # Drop multiple '\n\'s
            title = re.sub(r"\n", "", title)

            return title.lower()
        except Exception as e:
            logging.error(f"Couldn't get the title of url {url}, got error {str(e)}")

            return ""

    def label_url_content(self, csv_to_be_saved):
        """
        Strategy for labeling products from the pages

        For each link in the csv file:
            1. Get its title -> using BeautifulSoup or similar
            2. Split the text for that url into sentences
            3. Then split the resulted sentences into words
            4. Create a label of '0' (OUTSIDE) for the length of words list:
                4.1 If we got a word from the title -> label as 'B-PRODUCT'
                4.2 If the previous token label was 1 or 2 ('I-PRODUCT') and got a word from the title -> label as 'I-PRODUCT'
        """
        df = pd.read_csv(self.csv)

        urls = df['url'].tolist()
        contents = df['text'].tolist()
        labels = []

        for url, content in zip(urls, contents):
            labels_content = []
            logging.info(f"REACHED URL {url}\n...")

            title_url_words = self.grab_url_title(url).split()

            content_sentences = sent_tokenize(content)
            # content_sentences = [sentence.split("\n") for sentence in content_sentences if '\n' in sentence]
            # content_sentences = [sentence for sentence in content_sentences]

            logging.info(f"Found {len(content_sentences)} SENTENCES\n")

            # using IOB2 labeling scheme
            for idx, sentence in enumerate(content_sentences):
                words = sentence.lower().split()
                logging.info(f"Found WORDS {words} IN SENTENCE {idx + 1}")

                label = [0] * len(words)

                label = [1 if word in title_url_words else 0 for word in words]

                # Tried using list comprehension, but didn't really worked - TODO: try to fix this to work
                # if len(label) > 2:
                #     label = [2 if label[pos - 1] > 1 and word in title_url_words else label[pos]
                #              for word in words for pos in range(1, len(label))]

                if len(label) > 2:
                    for elem in range(1, len(label)):
                        # If the word is found in the title's word, and
                        #  we are still in the product name (either B-eginning or I-nside)
                        if words[elem] in title_url_words and label[elem - 1] > 0:
                            label[elem] = 2

                labels_content.extend(label)

            labels.append(labels_content)

        # print(labels)
        df['label'] = pd.Series(labels)  # add the new 'label' column to the dataset
        df.to_csv(path_or_buf=f"./{csv_to_be_saved}",
                  sep=',',
                  index=False)


def label_dataset():
    args = parser.parse_args()

    labeler = DatasetLabeller(args.path_to_csv)
    labeler.label_url_content(args.path_to_save_labeled_csv)


if __name__ == '__main__':
    label_dataset()


# Other attempts:

# ATTEMPT 1:
# # Tried using spacy's en_core_web_sm to automatically label the products
# - didn't really work, not many labelled PRODUCT
# import pandas as pd
# import spacy

# # Load the spaCy English language model
# nlp = spacy.load("en_core_web_sm")
#
# df = pd.read_csv("dataset_2.csv")
#
#
# # Function to automatically annotate texts with spaCy NER
# def spacy_annotation(text):
#     # Process the text with spaCy NER
#     doc = nlp(text)
#
#     # Convert entities to IOB2 format
#     iob2_labels = ['O'] * len(text.split())
#
#     try:
#         # Extract the entities and their labels
#         entities = [(ent.text, ent.label_) for ent in doc.ents]
#
#         # Convert entities to IOB2 format
#         for entity, label in entities:
#             words = entity.split()
#
#             if len(words) == 1:
#                 iob2_labels[doc.text.lower().split().index(words[0].lower())] = 'B-' + label
#             else:
#                 iob2_labels[doc.text.lower().split().index(words[0].lower())] = 'B-' + label
#
#                 for i in range(1, len(words)):
#                     iob2_labels[doc.text.lower().split().index(words[i].lower())] = 'I-' + label
#     except ValueError as e:
#         print(f"Error processing text: {text}. Error: {e}")
#
#     return ' '.join(iob2_labels)
#
#
# # Apply the spacy_annotation function to each row in the dataframe
# df['label'] = df['text'].apply(spacy_annotation)
#
# # Save the dataframe with the newly added 'label' column
# df.to_csv("labeled_dataset_2.csv", index=False)


# ###################################################################################################################


# ATTEMPT 2:
# Transformers - didn't worked
# import pandas as pd
# from transformers import pipeline
#
# # Load CSV file
# df = pd.read_csv("dataset_1.csv")
#
# # Load the zero-shot classification pipeline from transformers
# ner_pipeline = pipeline(task="ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
#
#
# # Function to automatically annotate texts with transformers NER
# def transformers_annotation(text):
#     # Use the transformers NER pipeline
#     entities = ner_pipeline([text])
#
#     # Convert entities to IOB2 format
#     iob2_labels = ['O'] * len(text.split())
#
#     for entity in entities[0]:
#         print(entity)
#
#         for i, word in enumerate(text.split()):
#             if word in entity["word"]:
#                 if i == entity["start"]:
#                     iob2_labels[i] = 'B-' + entity["entity"]
#                 elif i > entity["start"]:
#                     iob2_labels[i] = 'I-' + entity["entity"]
#
#     return ' '.join(iob2_labels)
#
#
# # Apply the transformers_annotation function to each row in the dataframe
# df['label'] = df['text'].apply(transformers_annotation)
#
# # Save the dataframe with the newly added 'label' column
# df.to_csv("labeled_data_transformers_1.csv", index=False)


# #################################################################################################################


# ATTEMPT 3
# Tried using a matcher with some rules for tagging the product. No success.
# import pandas as pd
# import spacy
# from spacy.matcher import Matcher
#
# nlp = spacy.load("en_core_web_sm")
# matcher = Matcher(nlp.vocab)
#
# # Define a pattern for product names
# furniture_patterns = [
#     [{"LOWER": {"IN": ["sofa", "couch", "settee", "loveseat"]}}],
#     [{"LOWER": {"IN": ["chair", "armchair", "recliner", "accent chair"]}}],
#     [{"LOWER": {"IN": ["table", "coffee table", "dining table", "side table"]}}],
#     [{"LOWER": {"IN": ["bed", "bedframe", "headboard", "mattress"]}}],
#     [{"LOWER": {"IN": ["cabinet", "wardrobe", "cupboard", "dresser"]}}]
# ]
#
# matcher.add("furniture_match", furniture_patterns)
#
#
# # Function to apply rule-based matching
# def rule_based_matching(text):
#     doc = nlp(str(text))
#     matches = matcher(doc)
#
#     iob2_labels = ['O'] * len(doc)
#
#     for match_id, start, end in matches:
#         iob2_labels[start] = 'B-PRODUCT'
#
#         for i in range(start + 1, end):
#             iob2_labels[i] = 'I-PRODUCT'
#
#     return ' '.join(iob2_labels)
#
# df = pd.read_csv("dataset.csv")
#
# # Apply the rule_based_matching function to each row in the dataframe
# df['label'] = df['text'].apply(rule_based_matching)
#
# # Save the dataframe with the newly added 'label' column
# df.to_csv("labeled_data_rule_based.csv", index=False)


# Attempt 4
# Tried using even more rules, with the IOBE labeling scheme instead.
"""
# import pandas as pd
# import spacy
# from spacy.matcher import Matcher
#
# # Load spaCy English language model
# nlp = spacy.load("en_core_web_sm")
#
# matcher = Matcher(nlp.vocab)
#
#
# # Define a pattern for product names
# furniture_patterns = [
#     [{"LOWER": {"IN": ["sofa", "couch", "settee", "loveseat", 'bench', 'ottoman', 'chaise lounge']}}],
#     [{"LOWER": {"IN": ["chair", "armchair", "recliner", "accent chair", 'futon', 'stool']}}],
#     [{"LOWER": {"IN": ["table", "coffee table", "dining table", "side table", 'desk',
#                        'bookshelf', 'nightstand', 'sideboard', 'entertainment center']}}],
#     [{"LOWER": {"IN": ["bed", "bedframe", "headboard", "mattress"]}}],
#     [{"LOWER": {"IN": ["cabinet", "wardrobe", "cupboard", "dresser"]}}]
# ]
#
# matcher.add("furniture_match", furniture_patterns)
#
#
# # Function to apply IOBE labeling scheme to a text with specific product names
# def label_text(text):
#     print("TEXT: ")
#     print(text)
#     print("\n\n")
#
#     doc = nlp(str(text))  # apply the model on the text
#     matches = matcher(doc)
#     labels = ['O'] * len(doc)  # Initialize labels as 'O' (Outside)
#
#     # for ent in doc.ents:
#     #     # Get the entities that are indeed labeled as products
#     #     if ent.label_ == 'PRODUCT':
#     #         start = ent.start
#     #         end = ent.end - 1
#     #         labels[start] = 'B-PRODUCT'
#     #
#     #         # For every label inside the entity name
#     #         # for i in range(start + 1, min(end + 1, len(labels))):
#     #         #     labels[i] = 'I-PRODUCT'
#     #
#     #         # Label the last position as the ending
#     #         if end < len(labels) - 1:
#     #             labels[end + 1] = 'E-PRODUCT'
#     #     print("LABELS")
#     #     print(labels)
#     #     print("\n\n")

#     # Maybe try the IOB2 scheme again
#     # iob2_labels = ['O'] * len(doc)
#
#     for match_id, start, end in matches:
#         labels[start] = 'B-PRODUCT'
#
#         for i in range(start + 1, end):
#             labels[i] = 'I-PRODUCT'
#
#     print(labels)
#     return ' '.join(labels)
#     # return labels
#
#
# # Read CSV file
# df = pd.read_csv('dataset_tuesday.csv')
#
# # Apply labeling function to each row
# df['label'] = df['text'].apply(label_text)
#
# # Save the updated dataframe to a new CSV file
# df.to_csv('labeled_dataset_tuesday.csv', index=False)
"""

