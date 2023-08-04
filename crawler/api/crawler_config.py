import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATASET_PATH = os.getenv("DATASET_RAW_PATH")
# DATASET_CSV = os.getenv('CSV_DATASET')
DATASET_CSV = "results_csv.csv"


@dataclass
class CrawlerConfig:
    proxies: Dict
    language: str
    featured_topics: List[str]
    auth: Union[str, None]
    resource_path: str
    repos_key_info: List[str]
    saved_repos: DataFrame
    wait_api: bool

    def __init__(
        self,
        resource_path,
        max_stars: Union[int, None],
        min_stars: Union[int, None],
        wait_api: bool = False,
        proxies: Dict = None,
        reload_features=True,
    ):
        if proxies is None:
            proxies = {
                "http": "http://proxy.jpmchase.net:10443",
                "https": "http://proxy.jpmchase.net:10443",
            }
        self.proxies = proxies
        self.language = "python"
        if reload_features:
            self.get_all_featured_github()
        with open(os.path.join("featured_topics.json"), "r") as f:
            self.featured_topics = json.load(f)
        self.additional_featured_topics = ["Computervision", "Reinforcementlearning"]
        self.featured_topics.extend(self.additional_featured_topics)
        self.resource_path = resource_path
        self.repos_key_info = [
            "full_name",
            "star_count",
            "language",
            "topics",
            "featured_topics",
            "added_on",
            "clone_url",
            "last_modified",
        ]
        self.auth = None
        self.wait_api = wait_api
        self.max_stars = max_stars
        self.min_stars = min_stars
        self.verify_params()
        self.saved_repos = self.read_csv()

    def read_csv(self) -> DataFrame:
        """
        Fetch an existing result CSV if it exists, else create and save it using the defined columns

        :return: CSV containing all the scrapped repositories information
        """
        csv_path = os.path.join(self.resource_path, DATASET_CSV)
        print(csv_path)
        if os.path.exists(csv_path):
            repo_infos = pd.read_csv(csv_path, index_col=0)
            if not (set(repo_infos.columns) == set(self.repos_key_info)):
                raise ValueError(
                    "Columns of the CSV do not correspond to configuration setting"
                )
        else:
            repo_infos = pd.DataFrame(columns=self.repos_key_info)
            repo_infos.to_csv(csv_path)
        return repo_infos

    def get_all_featured_github(self) -> None:
        """
        Get all GitHub featured topics

        :return: List of all GitHub featured topics
        """
        page = 1
        result = requests.get(
            f"https://github.com/topics/",
            proxies=None,
            params={"page": page},
            verify=False,
        )
        soup = BeautifulSoup(result.content, "html.parser")
        topics_section = soup.find_all("p", class_="f3")
        topics = [re.sub(r"\s+", "", topic.text) for topic in topics_section]
        while len(topics_section) > 0:
            page += 1
            result = requests.get(
                f"https://github.com/topics/",
                proxies=None,
                params={"page": page},
                verify=False,
            )
            soup = BeautifulSoup(result.content, "html.parser")
            topics_section = soup.find_all("p", class_="f3")
            topics.extend([re.sub(r"\s+", "", topic.text) for topic in topics_section])
        with open(os.path.join("featured_topics.json"), "w") as f:
            json.dump(topics, f)

    def verify_params(self):
        if not os.path.exists(self.resource_path):
            raise FileNotFoundError("This path does not exist")
        if self.min_stars > self.max_stars:
            raise ValueError(
                "Minimum number of stars can't be superior to maximum number of stars"
            )
