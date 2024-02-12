# As an extra step 6 I wanted to create a web page visualizer for the model results
import io
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_file
import json
import urllib.request
from urllib.parse import urljoin
import ssl
from nltk import sent_tokenize
from create_dataset import get_content_from_url
from predict import Predictor

app = Flask(__name__, template_folder=r'.\templates')


# Define app routes
# Main route:
@app.route('/')
def index():
    return render_template('index.html')


# Process URL route:
@app.route('/process-url', methods=['POST'])
def process_url():
    url = request.form.get('url')
    predictor = Predictor()
    all_products = {}

    if url:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urllib.request.urlopen(req, context=ssl.SSLContext()).read()
            soup = BeautifulSoup(html, 'lxml')

            links_in_url = soup.find_all('a')  # get all a-href
            # Get the page from the main url
            aggregate_pages = [get_content_from_url(url)]

            # Followed by pages from all links in the url
            for link in links_in_url:
                # Similar to what join in Python does, it combines a main url with the hyperlink -> absolute link
                link = urljoin(url, link.get('href'))

                link_content = get_content_from_url(link)
                aggregate_pages.append(link_content)

            pages = ''.join(aggregate_pages)
            split_sentences = sent_tokenize(pages)
            entities_sentences = []

            for sentence in split_sentences:
                entities_sentences.extend(predictor.predict_ner_page(sentence))

            all_products[url] = entities_sentences

            result_json = json.dumps(all_products, indent=4)

            return render_template('results.html', results=result_json)
        except Exception as e:
            return f"Got error {str(e)}"
    return "Got error: No URL provided to predict entities!"


# Download results as JSON route
@app.route('/download-json', methods=['POST'])
def download_json():
    # parse json data
    results = request.json.get('results')

    if results:
        json_output = json.dumps(results, indent=4)

        # create file obj
        output_json_file = io.BytesIO()
        output_json_file.write(json_output.encode('utf-8'))
        # move cursor at the beginning of the file
        output_json_file.seek(0)

        return send_file(
            output_json_file,
            mimetype='application/json',
            as_attachment=True,
            download_name='products_from_url.json'
        )
    return "Got error: no results to download as json!"


if __name__ == '__main__':
    app.run(debug=True)
