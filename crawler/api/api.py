import ast
import flask
import os
import tempfile
import json
import time
import pandas as pd
from typing import List, Dict, Union, Tuple, Any
from flask import request, jsonify, send_file, flash, render_template
from shutil import make_archive, rmtree
from crawler_config import CrawlerConfig
from github_crawler import GithubCrawler

app = flask.Flask(__name__)
app.config['DEBUG'] = True

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
API_DIR = os.getenv('API_PATH')
TOPICAL_PATH = os.getenv('TOPICAL_PATH')
CSV_DATASET = os.getenv('CSV_DATASET')
DATASET_RAW_PATH = os.getenv('DATASET_RAW_PATH')

@app.route('/', methods=['GET', 'POST'])
def dataframe_to_html():
    dataframe = pd.read_csv(os.path.join(ROOT_DIR, TOPICAL_PATH, DATASET_RAW_PATH, CSV_DATASET))
    dataframe_html = dataframe.to_html(classes='data')
    return render_template(os.path.join(ROOT_DIR, API_DIR,'templates\\download.html'), output=dataframe_html)

@app.route('/api/', methods=['POST'])
def run_crawler():
    """
    POST Method to be run in the HTML app to upload and download using a form

    :return: zip file containing the csv description and repository folders
    """
    """if request.files['file'].name.endswith('.json'):
        topics = json.load(request.files['file'])
    else:
        raise TypeError('Only JSON can be provided')"""
    topics = json.load(request.files['file'])
    valid, topics = check_topics_validity(topics)
    proxies = {'http': 'http://proxy.jpmchase.net:10443','https':'http://proxy.jpmchase.net:10443'}

    if not valid:
        raise ValueError('Provided JSON file is not valid')
    with tempfile.TemporaryDirectory() as dirpath:
        crawler_config = CrawlerConfig(dirpath, 30, 0, False,proxies=proxies)
        crawler = GithubCrawler(crawler_config)
        if topics is not None:
            crawler.scrap(topics, limit=2)
        f= tempfile.TemporaryDirectory()
        tmparchive = os.path.join(f.name, 'scrapped_repositories')
        make_archive(tmparchive, 'zip', dirpath)

    return send_file(str(tmparchive)+'.zip', as_attachment=True)

@app.route('/run/<topics>', methods=['GET'])
def run_crawler_cmd(topics):
    """
    GET Method to be used in command line or in the browser

    :param topics: List or dict of topics
    :return:  zip file containing the csv description and repository folders
    """
    topics = ast.literal_eval(topics)
    valid, topics = check_topics_validity(topics)
    proxies = {'http': 'http://proxy.jpmchase.net:10443', 'https': 'http://proxy.jpmchase.net:10443'}

    with tempfile.TemporaryDirectory() as dirpath:
        crawler_config = CrawlerConfig(dirpath, 30, 0, False, proxies=proxies)
        crawler = GithubCrawler(crawler_config)
        if topics is not None:
            crawler.scrap(topics, limit=1)
        f= tempfile.TemporaryDirectory()
        tmparchive = os.path.join(f.name, 'scrapped_repositories')
        make_archive(tmparchive, 'zip', dirpath)

    return send_file(str(tmparchive)+'.zip', as_attachment=True)

def check_topics_validity(topics) -> Tuple[bool, Any]:
    """
    Check the validity of the topics whether it is a list, dict or string

    :param topics: object containing the topic information
    :return: Tuple containing a boolean indicating the validity of the object and the object itself
    """

    # Check the type of the object, if it is a dictionnary, check that this dictionnary only contains Dict, List or str
    if isinstance(topics, List) and all([isinstance(e, str) for e in topics]):
        check=True
    elif isinstance(topics, List):
        check=False
    elif isinstance(topics, Dict):
        try:
            flatten_dict = pd.json_normalize(topics).to_dict(orient='records')[0]
            if all([isinstance(e, str) for e in list(flatten_dict.values())]):
                check=True
            else:
                check=False
        except:
            check=False
    else:
        check=False

    return check, topics

app.secret_key = 'secret key'
app.run()

