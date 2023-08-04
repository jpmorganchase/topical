import pytest
import os
import builtins
import io
import sys
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
API_DIR = os.getenv('API_PATH')
TEST_DIR = os.getenv('TEST_DIR')
sys.path.append(os.path.join(ROOT_DIR, API_DIR))
from crawler_config import CrawlerConfig
from github_crawler import GithubCrawler
from shutil import rmtree

@pytest.mark.parametrize(
    "config_file, output, max_stars, min_stars",
    [
        (os.path.join(TEST_DIR, 'test_csv_wrong'), ValueError, 10, 5),
        (os.path.join(TEST_DIR,'test_csv_right'), None, 10, 5),
        (os.path.join(TEST_DIR,'tst_csv_right'), FileNotFoundError, 10, 5),
        (os.path.join(TEST_DIR,'test_csv_right'), ValueError, 5, 10)
    ]
)

def test_get_config(config_file, output, max_stars, min_stars):
    if type(output) == type and issubclass(output, Exception):
        with pytest.raises(output):
            crawler_config = CrawlerConfig(config_file, max_stars = max_stars, min_stars = min_stars)
    else:
        crawler_config = CrawlerConfig(config_file, max_stars = max_stars, min_stars = min_stars)
        assert len(crawler_config.featured_topics) > 0
        assert crawler_config.max_stars == max_stars
        assert crawler_config.min_stars == min_stars

@pytest.fixture(scope="function")
def github_crawler():
    return GithubCrawler(CrawlerConfig(os.path.join(TEST_DIR,'test_csv_right'), max_stars=10, min_stars=5))

"""@pytest.fixture
def delete_temporary_dataset(self):
    os.remove('test_dataset')
    return None"""

@pytest.mark.parametrize(
    "topics, limit, output",
    [('CV', 1, None),
     (None, 1, None),
     (["ComputerVision", "NLP"], 1, None),
     ({'MachineLearning': ['NLP']}, 1, None),
     (10, 0, TypeError),
     (None, 'abc', TypeError)
    ]
)
def test_scrap(topics, limit, output, github_crawler):
    if type(output) == type and issubclass(output, Exception):
        with pytest.raises(output):
            github_crawler.scrap(topics, limit)
    else:
        github_crawler.scrap(topics, limit)
        if isinstance(topics, list):
            assert len(os.listdir(github_crawler.config.resource_path)) == (limit+1)*len(topics) + 1
        elif isinstance(topics, dict):
            assert len(os.listdir(github_crawler.config.resource_path)) == (limit+1)*sum([len(t) for t in topics.values()]) + 1
        else:
            assert len(os.listdir(github_crawler.config.resource_path)) == limit+2
        for doc in os.listdir(github_crawler.config.resource_path):
            if os.path.isdir(os.path.join(github_crawler.config.resource_path, doc)):
                rmtree(os.path.join(github_crawler.config.resource_path, doc))
            else:
                os.remove(os.path.join(github_crawler.config.resource_path, doc))



@pytest.mark.parametrize(
    "topics, featured_topics",
    [(['CV'], ["Computervision"]),
     (['Computer-vision'], ["Computervision"]),
     (['company'], [])
     ]
)
def test_match_featured_topics(topics, featured_topics, github_crawler):
    f_topics = github_crawler.match_topics_to_featured(topics, 90)
    assert f_topics == featured_topics




