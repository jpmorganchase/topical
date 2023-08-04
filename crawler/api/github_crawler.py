import argparse
import ast
import json
import logging
import os
import platform
import re
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process
from requests import Response
from tqdm import tqdm
from .crawler_config import CrawlerConfig

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATASET_PATH = os.getenv("DATASET_RAW_PATH")
DATASET_CSV = os.getenv("CSV_DATASET")
API_PATH = os.getenv("API_PATH")


class GithubCrawler:
    def __init__(self, config: CrawlerConfig) -> None:
        """
        Crawler for scrapping repositories and their metadata from GitHub given specific topics

        :param config: Pre-defined configuration for the crawler
        """
        self.config = config
        self.saved_repos = self.config.saved_repos

    def save_csv(self) -> None:
        """
        Save the updated result CSV at the indicated path

        """
        csv_path = os.path.join(self.config.resource_path, DATASET_CSV)
        self.saved_repos.to_csv(csv_path)

    def scrap(
        self, topics: Union[List, str, Dict, None], limit: Union[int, None]
    ) -> None:
        """
        Factory to scrap either indicated topics or random topics

        :param topics: list, dictionnary or string containing all the topics to be scrapped
        :param limit: Number of repositories to be scrapped per topic
        """
        if topics is not None:
            self.scrap_topics(topics, limit)
        else:
            self.scrap_all_topics(limit)

    def scrap_topics(
        self, topics: Union[List, str, Dict], limit: Union[int, None] = None
    ) -> None:
        """
        Call the REST GitHub Api to fetch repositories information according to the desired topics

        :param topics: list, dictionnary or string containing all the topics to be scrapped
        :param limit: Number of repositories to be scrapped per topic
        """
        counter = 0
        page = 1
        status = 200
        if isinstance(topics, Dict):
            self.make_tree_request(topics, limit)
        elif isinstance(topics, List):
            for topic in tqdm(topics):
                status = 200
                counter = 0
                page = 0
                while status == 200:
                    r = self.request_api(topics=topic, page=page)
                    if (
                        ((limit is not None and counter < limit) or limit is None)
                        and r.status_code == 200
                        and "items" in r.json().keys()
                    ):
                        page += 1
                        counter += self.update_csv_results_various_topics(
                            r.json()["items"], limit
                        )
                        status = r.status_code
                    else:
                        status = 300
        elif isinstance(topics, str):
            while status == 200:
                r = self.request_api(topics=topics, page=page)
                if (
                    ((limit is not None and counter < limit) or limit is None)
                    and r.status_code == 200
                    and "items" in r.json().keys()
                ):
                    page += 1
                    counter += self.update_csv_results_various_topics(
                        r.json()["items"], limit
                    )
                    status = r.status_code
                else:
                    status = 300
        logging.info("Search successfully completed")

    def scrap_all_topics(self, limit: Union[int, None] = None) -> None:
        """
        Scrap random topics from Github using parameters set in the config and as arguments to the crawler

        :param limit: number of total repositories to be scrapped
        """
        i = 1
        counter = 0
        status = 200
        while status == 200:
            r = self.request_api(topics=None, page=i)
            if (
                (limit is not None and counter < limit)
                or limit is None
                and r.status_code == 200
                and "items" in r.json().keys()
            ):
                i += 1
                nb_results = self.update_csv_results_various_topics(
                    r.json()["items"], limit
                )
                counter += nb_results
                status = r.status_code
            else:
                continue
        logging.info("Search successfully completed")

    def scrap_topics_github_page(self, full_name: str) -> List[str]:
        """
        Scrap all the topics displayed on a given GitHub page

        :param full_name: Name of the repositories to scrap: usename/repo_name
        :return: list of topics for a given repository
        """
        result = requests.get(
            f"https://github.com/{full_name}", proxies=self.config.proxies, verify=False
        )
        soup = BeautifulSoup(result.content, "html.parser")
        topics_section = soup.find_all("a", class_="topic-tag")
        topics = [re.sub(r"\s+", "", topic.text) for topic in topics_section]
        return topics

    def update_csv_results_various_topics(
        self, api_results: List[Dict], limit: Union[int, None]
    ) -> int:
        """
        For each repositories info contained in the GitHub Reponse,
        verify it is not scrapped already then scrap it and add it to the recapitulative CSV

        :param api_results: List of reponses from the GitHub API
        :param limit: Number of repositories to scrap
        :return: Number of repositories scrapped
        """
        counter = 0
        for result in api_results:
            print(result)
            if limit is not None and counter > limit:
                logging.info(f"{counter} new repositories were added to the dataset")
                return counter
            elif (
                result["full_name"].replace("/", "_")
                not in self.saved_repos["full_name"].values
                and result["fork"] == False
            ):
                topics = self.scrap_topics_github_page(result["full_name"])
                featured_topics = self.match_topics_to_featured(topics, 90)
                repo_info = {
                    "full_name": result["full_name"].replace("/", "_"),
                    "star_count": result["stargazers_count"],
                    "language": self.config.language,
                    "topics": topics,
                    "featured_topics": featured_topics,
                    "added_on": datetime.now(),
                    "last_modified": datetime.now(),
                    "clone_url": result["clone_url"],
                }

                code_status = self.download_repository(
                    repo_info["full_name"], result["full_name"]
                )
                if code_status == 200:
                    counter += 1
                    self.saved_repos = self.saved_repos.append(
                        repo_info, ignore_index=True
                    )
                    self.save_csv()
            else:
                pass
        logging.info(f"{counter} new repositories were added to the dataset")
        return counter

    def request_api(
        self, topics: Union[str, List, None], page: Union[int, None] = None
    ) -> Response:
        """
        Makes a request to the GitHub API using defined parameters

        :param topics: List of topics to scrap
        :param page: number of the page of results to fetch
        :return: GitHub API Reponse
        """
        url_request = f"https://api.github.com/search/repositories?q=is:featured+language:{self.config.language}"

        if self.config.max_stars is not None:
            url_request += f"+stars:<={self.config.max_stars}"
        elif self.config.min_stars is not None:
            url_request += f"+stars:>={self.config.min_stars}"
        if isinstance(topics, str):
            url_request += f"+topic:{topics.strip()}"

        elif isinstance(topics, List):
            for topic in topics:
                url_request += f"+topic:{topic}"
            print(url_request)

        elif topics is None:
            pass

        else:
            raise TypeError(f"Expected list or string type, got {type(topics)} instead")

        if self.config.auth is not None:
            headers = {"Authorization": self.config.auth}

        else:
            headers = None
        if page is not None:
            payload = {"per_page": 100, "page": page}
        else:
            payload = {"per_page": 100}
        r = requests.get(
            url_request,
            headers=headers,
            proxies=self.config.proxies,
            params=payload,
            verify=False,
        )
        return r

    def download_repository(self, repository_name: str, repo_name: str) -> int:
        """
        Downloading a repository from its GitHub link

        :param repository_name: local name
        :param repo_name: Github link : username/repo_name
        :return: status code of the request
        """

        repository_path = os.path.join(self.config.resource_path, repository_name)
        if not (os.path.exists(repository_path) and os.path.isdir(repository_path)):
            os.mkdir(repository_path)
            r = requests.get(
                f"https://api.github.com/repos/{repo_name}/git/trees/master",
                proxies=self.config.proxies,
                params={"recursive": "true"},
                verify=False,
            )
            if r.status_code != 200 and r.status_code != 404 and self.config.wait_api:
                logging.info("API calls limit reached: sleep for 1 hour. ZzZzZzz...")
                time.sleep(3600)
                self.download_repository(repository_name, repo_name)
            elif r.status_code != 200 and r.status_code != 404:
                logging.info("API calls limit reached. Ending scrapping")
                exit(0)
            elif r.status_code == 404:
                return r.status_code
            else:
                tree = r.json()["tree"]
                paths = [
                    t["path"]
                    for t in tree
                    if t["type"] == "blob" and t["path"].endswith(".py")
                ]
                # or fuzz.partial_ratio(t['path'].lower(), "requirements.txt") > 90
                # or fuzz.partial_ratio(t['path'].lower(), "readme.txt") > 90)
                folders = [t["path"] for t in tree if t["type"] == "tree"]
                self.create_folders(folders, repository_path)
                self.create_files(paths, repository_path, repo_name)
                logging.info(
                    f"{repository_name} successfully downloaded into {repository_path}"
                )
            return r.status_code
        else:
            logging.info(f"{repository_name} already exists in {repository_path}")
            return 400

    def make_tree_request(
        self, dict_request: Dict, limit_number_by_topic: Union[int, None]
    ) -> None:
        """
        Download all repositories given a dictionnary containing tree requests

        :param dict_request: Dictonnary containing the topics to scrap
        :param limit_number_by_topic: Number of repositories to scrap per branch
        :return: status code of the request
        """
        counter = 0
        page = 1
        status = 200
        normalised_request = pd.json_normalize(dict_request, sep="/").to_dict(
            orient="records"
        )[0]
        for key, item in normalised_request.items():
            request = key.split("/")
            if isinstance(item, List):
                for i in item:
                    topics = request + [i]
                    status = 200
                    counter = 0
                    while status == 200:
                        r = self.request_api(topics=topics, page=page)
                        if (
                            (
                                limit_number_by_topic is not None
                                and counter < limit_number_by_topic
                            )
                            or limit_number_by_topic is None
                            and r.status_code == 200
                        ):
                            page += 1
                            counter += self.update_csv_results_various_topics(
                                r.json()["items"], limit_number_by_topic
                            )
                            status = 200
                        else:
                            status = 300
            elif isinstance(item, str):
                topics = request + [item]
                while status == 200:
                    r = self.request_api(topics=topics, page=page)
                    if (
                        (
                            limit_number_by_topic is not None
                            and counter < limit_number_by_topic
                        )
                        or limit_number_by_topic is None
                        and r.status_code == 200
                    ):
                        page += 1
                        counter += self.update_csv_results_various_topics(
                            r.json()["items"], limit_number_by_topic
                        )
                    else:
                        status = 300

    def get_stats(self) -> None:
        if "stats" not in os.listdir(DATASET_PATH):
            os.mkdir(os.path.join(DATASET_PATH, "stats"))
        else:
            pass

        timestamp = time.strftime("%Y%m%d-%H%M")
        featured_topics = []
        # read_mes = []
        for i, row in self.saved_repos.iterrows():
            # read_mes.append(self.search_readmes(os.path.join('dataset', row['full_name'])))
            featured_topics.append(self.match_topics_to_featured(row["topics"], 80))

        featured_topics = [topic for topics in featured_topics for topic in topics]
        topics = [
            t
            for topic in self.saved_repos["topics"].tolist()
            for t in ast.literal_eval(topic)
        ]
        len_topics = [
            len(ast.literal_eval(topic))
            for topic in self.saved_repos["topics"].tolist()
        ]

        # STATS ON FEATURED TOPICS
        count_featured = Counter(featured_topics)
        most_common = count_featured.most_common(20)
        x = [t[0] for t in most_common]
        y = [t[1] for t in most_common]
        plt.figure(figsize=(30, 10))
        plt.xticks(fontsize=30, rotation=90)
        plt.yticks(fontsize=25)
        plt.bar(x, y)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config.resource_path, "stats", f"featured_topics_{timestamp}.png"
            )
        )

        # STATS ON USER-HAND PICKED TOPICS
        counter_topics = Counter(topics)
        most_common = counter_topics.most_common(20)
        x = [t[0] for t in most_common]
        y = [t[1] for t in most_common]
        plt.figure(figsize=(30, 10))
        plt.xticks(fontsize=30, rotation=90)
        plt.yticks(fontsize=25)
        plt.bar(x, y)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config.resource_path, "stats", f"raw_topics_{timestamp}.png"
            )
        )

        # STATS NUMBER OF TOPICS BY REPO
        plt.hist(len_topics, range=(0, 20), bins=21)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config.resource_path, "stats", f"number_topics_{timestamp}.png"
            )
        )

    def match_topics_to_featured(
        self, topics: Union[List[str], str], threshold: int
    ) -> List[str]:
        """
        Match hand-picked topics with GitHub featured-topics

        :param topics: List of topics to be matched
        :param threshold: threshold of similarity for matching
        :return: List of matched featured-topics
        """
        abv_topics = {
            "nlp": "Naturallanguageprocessing",
            "cv": "Computervision",
            "rl": "Reinforcementlearning",
        }
        try:
            topics = ast.literal_eval(topics)
        except:
            pass
        topics = [
            abv_topics[topic.lower()] if topic.lower() in abv_topics.keys() else topic
            for topic in topics
        ]
        matches = [
            process.extract(topic, self.config.featured_topics)[0][0]
            for topic in topics
            if (
                process.extract(topic, self.config.featured_topics)[0][1] > threshold
                and len(topic) > 2
            )
            or (process.extract(topic, self.config.featured_topics)[0][1] == 100)
        ]
        return matches

    def create_files(
        self, paths: List[str], repository_path: str, repo_name: str
    ) -> None:
        """
        Download the python files for a given repository

        :param paths: all the files path in the repository
        :param repository_path: local path of the repository
        :param repo_name: repository name : username/repo_name
        """
        headers = {"connection": "keep-alive", "keep-alive": "timeout=10, max=1000"}
        s = requests.Session()
        s.proxies.update(self.config.proxies)
        for path in paths:
            if platform.system() == "Windows":
                file_path = path.replace("/", "\\")
            else:
                file_path = path
            with open(os.path.join(repository_path, file_path), "wb") as f:
                r = s.get(
                    f"https://github.com/{repo_name}/blob/master/{path}",
                    proxies=self.config.proxies,
                    params={"raw": "true"},
                    headers=headers,
                    verify=False,
                )
                f.write(r.content)

    def get_number_of_commits(self, repository_name: str) -> Tuple[Dict, int]:
        """
        For each first-level folder in a repository, get the number of associated commits

        :param repository_name: Name of the repository
        :return: Tuple containing a dictionnary with the folder names as keys and number of commits as values
        and an integrer indicating the depth in the repository
        """
        headers = {"connection": "keep-alive", "keep-alive": "timeout=10, max=1000"}
        folder_path = os.path.join(self.config.resource_path, repository_name)
        folders = os.listdir(folder_path)
        depth = 0
        while len(folders) < 2:
            if len(folders) == 0 or os.path.isfile(folders[0]):
                return {}, -1
            else:
                folder_path = os.path.join(folder_path, folders[0])
                folders = os.listdir(folder_path)
                depth += 1
        folder_count = {}
        repo_name = repository_name.replace("_", "/", 1)
        for folder in folders:
            folder = os.path.join(*folder_path.split(os.sep)[2:], folder)
            commits = []
            request = requests.get(
                f"https://github.com/{repo_name}/commits",
                proxies=self.config.proxies,
                params={"path": folder, "branch": "master"},
                headers=headers,
                verify=False,
            )
            soup = BeautifulSoup(request.content, "html.parser")
            scrapped_commits = soup.find_all("li", class_="js-commits-list-item")
            for commit in scrapped_commits:
                commits.append(
                    re.findall(
                        r"(?<=commit\/).*(?=\/_render_node)", commit.get("data-url")
                    )[0]
                )
            if len(commits) == 0:
                print(repo_name)
                print(folder)
            last_commit = commits[-1]
            while len(scrapped_commits) > 0:
                request = requests.get(
                    f"https://github.com/{repo_name}/commits?path={folder}&branch=master&after={last_commit}+0".encode(
                        "utf-8"
                    ),
                    proxies=self.config.proxies,
                    verify=False,
                )
                soup = BeautifulSoup(request.content, "html.parser")
                scrapped_commits = soup.find_all("li", class_="js-commits-list-item")
                for commit in scrapped_commits:
                    commit_tag = re.findall(
                        r"(?<=commit\/).*(?=\/_render_node)", commit.get("data-url")
                    )[0]
                    commits.append(commit_tag)
                last_commit = commits[-1]

            folder_count[folder] = len(commits)
        return folder_count, depth + 2

    def get_all_featured_github(self):
        """
        Get all the GitHub featured topics
        """
        page = 1
        result = requests.get(
            f"https://github.com/topics/",
            proxies=self.config.proxies,
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
                proxies=self.config.proxies,
                params={"page": page},
                verify=False,
            )
            soup = BeautifulSoup(result.content, "html.parser")
            topics_section = soup.find_all("p", class_="f3")
            topics.extend([re.sub(r"\s+", "", topic.text) for topic in topics_section])
        return topics

    @staticmethod
    def test_auth(auth: str) -> Union[str, None]:
        """Test if the given credentials are correct, else connect to the API without credentials"""
        headers = {"Authorization: token": auth}
        r = requests.get("https://api.github.com", headers=headers, verify=False)
        status_code = r.status_code
        if status_code != 200:
            logging.warning(
                f"Authentification was not successful: error {status_code}, request without authentification"
            )
            return None
        else:
            logging.info("Authentification was successful")
            return auth

    @staticmethod
    def create_folders(folders: List[str], repository_path: str) -> None:
        """
        Create the structure of the repository to download

        :param folders: path of the repository to create
        :param repository_path: path of the repository to fill
        """
        for subfolder in folders:
            if platform.system() == "Windows":
                subfolder = subfolder.replace("/", "\\")
            os.mkdir(os.path.join(repository_path, subfolder))


def search_readmes(repo_path: str) -> bool:
    """
    Search through the repository for a README file

    :param repo_path: Path of the repository to inspect
    :return: Boolean to indicate the presence of a README
    """
    for dirpath, dir_name, filename in os.walk(repo_path):
        for file in filename:
            if fuzz.partial_ratio(file.lower(), "readme") > 80:
                return True
    return False


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of repositories to download, if topics are indicated then it is the maximum number per topic",
        default=None,
    )
    parser.add_argument(
        "--topics",
        type=str,
        help="Specific topics to scrap, can be a single one or a json file containing a more complex structure. See doc",
        default=None,
    )
    parser.add_argument(
        "--run_stats",
        action="store_true",
        help="Permits to run a complete report of the actual dataset composition",
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        default="dataset",
        type=str,
        help="Directory in which the repositories will be saved",
    )
    parser.add_argument(
        "--wait_api",
        action="store_true",
        help="Set a sleep timer when API limit is reached so the script continues to run, else finish the job",
        default=False,
    )
    parser.add_argument("--max_stars", type=int, default=50)
    parser.add_argument("--min_stars", type=int, default=None)
    parser.add_argument(
        "--proxies",
        type=str,
        default="['http://proxy.jpmchase.net:10443','http://proxy.jpmchase.net:10443']",
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    proxies = list(map(str, args.proxies.strip("[]").split(",")))
    crawler_config = CrawlerConfig(
        args.output_dir,
        args.max_stars,
        args.min_stars,
        args.wait_api,
        {"http": proxies[0], "https": proxies[1]},
    )
    crawler = GithubCrawler(crawler_config)
    if args.run_stats:
        crawler.get_stats()
    else:
        topics = args.topics
        if topics is not None and topics.endswith(".json"):
            with open(topics, "r") as f:
                topics = json.load(f)
        crawler.scrap(topics, args.limit)
