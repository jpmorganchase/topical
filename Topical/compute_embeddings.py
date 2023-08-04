"""
This script permits to compute the different embeddings composing Topical hybrid Script Embeddings
"""
import argparse
import ast
import json
import os
import random
import re
import subprocess
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATASET_PATH = os.getenv("DATASET_RAW_PATH")
OUTPUT_PATH = "output"
DATASET_CSV = os.getenv("CSV_DATASET")
ENGLISH_MODEL = os.getenv("ENGLISH_MODEL")
CODE_MODEL = os.getenv("CODE_MODEL")
RESOURCES_PKL = os.getenv("RESOURCES_PKL")
TOPICAL_PATH = os.getenv("TOPICAL_PATH")
warnings.filterwarnings("ignore")

## OPEN THE REPOSITORIES DATAFRAME
print(ENGLISH_MODEL)

## INSTANCIATE BERT BASES AND THEIR ASSOCIATED TOKENIZERS
tokenizers = {
    "code": RobertaTokenizer.from_pretrained(CODE_MODEL),
    "english": DistilBertTokenizer.from_pretrained(ENGLISH_MODEL),
}
models = {
    "code": RobertaModel.from_pretrained(CODE_MODEL),
    "english": DistilBertModel.from_pretrained(ENGLISH_MODEL),
}

tokenizers["english"].add_special_tokens({"additional_special_tokens": ["[C]"]})
models["english"].resize_token_embeddings(len(tokenizers["english"]))

def get_modules(file_content: str) -> Tuple[List[str], List[str]]:
    """
    Gets the imports and function names for a given python script content

    :param file_content: Python script content
    :return: A tuple containing the list of imports and the list of function names
    """
    imports = set()
    functions = set()
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return imports, functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for subnode in node.names:
                imports.add(subnode.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.add(node.module)
        elif isinstance(node, ast.FunctionDef):
            functions.add(node.name)

    return imports, functions


def get_first(s: str, depth=None):
    """
    Function to separate the different components of an import name (path materialized by a '.')

    :param s: raw import name
    :param depth: number of components of the raw name to extract (from the deepest module to the largest one)
    :return: converted import name
    """
    if os.sep in s:
        s = s.split(os.sep)[-1]
    else:
        pass
    if "<builtin>" in s:
        return ""
    elif "." in s and depth is None:
        return " ".join(s.split("."))
    elif "." in s and depth > len(s.split(".")):
        return " ".join(s.split(".")[:depth])
    elif "." in s:
        return " ".join(s.split("."))
    else:
        return s


def complete_dataset_ast(file_content: str) -> Tuple[str, str]:
    """
    Get all the children and parents for a given file content, concatenate children together and parents together

    :param file_content: Python script content
    :return: Tuple containing a string of all parents and a string of all children for a given script content
    """
    parents, children = get_modules(file_content)
    output = (
        " ".join(
            [
                get_first(parent)
                for parent in list(parents)
                if get_first(parent) is not None
            ]
        ),
        " ".join(list(children)),
    )
    return output


def get_docstring(file_content: str) -> Union[str, None]:
    """
    Get all the docstring and function names from a given Script and concatenate
    them separing both type of data by a special token [SEP]

    :param file_content: Python script content
    :return: String of concatenated docstrings and function names for the given script content, None if the script didn't contain any
    """
    found_exception = False
    try:
        tree = ast.parse(file_content)
    except Exception as e:
        print(e)
        found_exception = True
        docstring = ""
        return docstring

    functions = [
        re.sub(r"_+", " ", f.name).strip()
        for f in tree.body
        if isinstance(f, ast.FunctionDef)
    ]
    functions = [" ".join(re.findall("[^A-Z]*", f)) for f in functions]
    if len(functions) > 30:
        functions = functions[0:30]

    docstring = [
        re.sub(r"\s", " ", ast.get_docstring(f))
        for f in tree.body
        if isinstance(f, ast.FunctionDef) and ast.get_docstring(f) is not None
    ]
    # maximum docstring 100 chars
    docstring = " ".join([doc[0:100] if len(doc) > 100 else doc for doc in docstring])

    if docstring is None:
        docstring = ""
    else:
        pass
    print(len(docstring))
    if len(docstring) > 250:
        docstring = docstring[0:250]

    if len(functions) > 0 and len(docstring) > 0:
        docstring = " ".join(functions) + " [SEP] " + docstring
    elif len(functions) > 0:
        docstring = " ".join(functions)
    else:
        pass
    docstring = re.sub("  ", " ", docstring)
    return docstring.strip()


def iter_json(
    data_d: Dict, pycg_list: List[Tuple], gen: int = 0, parent: List = []
) -> None:
    """
    Recursive function to populate parent - child edges from PyCG dependencies graph

    :param data_d: children left to parse (fraction of PyCG dependencies graph left to parse)
    :param pycg_list: parent-child edges
    :param gen: current depth of exploration of the depedencies graph
    :param parent: List of parents that have been parsed
    """
    if isinstance(data_d, dict):
        keys = list(data_d.keys())
        for k in keys:
            parent = get_first(parent, 2)
            k0 = get_first(k, 2)
            if len(parent) > 0 and len(k0) > 0:
                pycg_list.append((parent, k0))
            iter_json(data_d[k], pycg_list, gen + 1, k)
    if isinstance(data_d, list):
        for l in data_d:
            iter_json(l, pycg_list, gen, parent)
    if isinstance(data_d, str):
        parent = get_first(parent, 2)
        data_d = get_first(data_d, 2)
        if len(parent) > 0 and len(data_d) > 0:
            pycg_list.append((parent, data_d))


def get_pycg(file_path: str, path) -> Union[Dict, None]:
    """
    PyCG command line to generate a temporary json file containing the dependency graph for a given file

    :param file_path: Path to the file
    :return: Dict containing the dependency graph for a given file
    """
    try:
        root = os.path.dirname(file_path)
        subprocess.run(
            "pycg --package {Root} {file_path} -o temporary_output.json".format(
                file_path=file_path, Root=root
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        with open("C:\JPMC\DEV\TMP\ds\Topical\senatus-code-s4\examples\\temporary_output.json", "rb") as f:
            data = json.load(f)
        os.remove("C:\JPMC\DEV\TMP\ds\Topical\senatus-code-s4\examples\\temporary_output.json")
    except Exception as err:    
        print('hii',err,file_path,)
        return None

    return data


def complete_dataset_pycg(file: str, path: str) -> List[str]:
    """
    Call PyCG parser and get dependencies from the raw graph

    :param file: Path to the file to parse
    :return: List of tuple (parent, children)
    """
    output = get_pycg(file, path)
    print("Output: {}".format(output))
    parse_pycg = []
    iter_json(output, parse_pycg)
    if len(parse_pycg) > 30:
        return parse_pycg[0:30]
    return parse_pycg


def read_script(script_path: str) -> Union[str, None]:
    """
    Returns a script given a path if valid

    :param script_path: Path to the script to read
    :return: Script content, None if the given path is not valid
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
    except:
        return None
    return content


def get_embeddings(input_seq: List[Dict], type: str = "code") -> np.array:
    """
    Returns the embeddings for a list of tokenized elements.

    :param input_seq: List of tokenized code using the appropriate tokenizer
    :param type: Type of element to embed, possible values = ['code', 'english']
    :return: Matrix of embeddings corresponding
    """
    embeddings = []
    if type not in models.keys():
        raise ValueError("type of embedding not supported")
    else:
        with torch.no_grad():
            for inp in input_seq:
                inp['input_ids'] = inp['input_ids'][:,:512]
                inp['attention_mask'] = inp['attention_mask'][:,:512]
                output = models[type](**inp)
                embeddings.append(output[0][:, 0, :].detach().numpy())
        embeddings = torch.tensor(embeddings)
        if len(embeddings.shape) > 1:
            embeddings = np.array(embeddings.squeeze(1))
        else:
            embeddings = np.array(embeddings)
    return embeddings


def get_tokens(
    input_seq: List[str], type: str, type_dep: Union[str, None] = None
) -> List[Dict]:
    if type not in tokenizers.keys():
        raise ValueError("input type not supported")
    else:
        if type == "english" and type_dep == "pycg":
            input_seq = [
                "[SEP]".join(
                    [f'{" ".join(t[0].split(os.sep)[2:])} [C] {t[1]}' for t in ele]
                )
                for ele in input_seq
            ]
        elif type == "english" and type_dep == "ast":
            input_seq = [f"{t[0]} [SEP] {t[1]}" for t in input_seq]

    return input_seq


def explore_repository(repo_path: str) -> List[str]:
    """
    Returns a list of all python script paths contained in a griven repository

    :param repo_path: Path to the repository to explore
    :return: List of Python script paths
    """

    def ignore(check_path):
        if "site-packages" in check_path:
            return True
        if "__init__" in check_path:
            return True
        if "lib" in check_path:
            return True
        return False

    repo_paths = []
    for dir_path, dirname, filename in os.walk(repo_path):
        if ignore(dir_path):
            continue
        if ignore(dirname):
            continue
        for file in filename:
            if ignore(file):
                continue
            if file.endswith(".py"):
                repo_paths.append(os.path.join(dir_path, file))
    return repo_paths


def choose_file(repo_files, dep_graph=None):
    if dep_graph == None:
        return repo_files.pop(0)
    else:
        files_i = []
        for line in dep_graph:
            files_i.extend(
                [
                    i
                    for i in range(len(repo_files))
                    if is_child_in_file(line[1], repo_files[i])
                ]
            )
        files_i = list(set(files_i))
        files_i.sort()
        print("found files:", files_i)
        for i in files_i:
            repo_files.insert(0, repo_files.pop(i))

        return repo_files.pop(0)


def is_child_in_file(child, file):
    name = file.split("/")[-1]
    name = re.sub(".py", "", name)
    if (
        name in child.split(" ")[:-1]
    ):  # exclude the last one because it can't be a file name but a method/object name.
        return True
    else:
        return False


def compute_resources(dataset: pd.DataFrame, path, pycg: bool = True,num_sampling_scripts: int = 15) -> pd.DataFrame:
    """
    Completed the Dataframe containing Dataset repository info with computed embeddings for each domains of the Hydrid
    Script-level embedding used by Topical.

    :param dataset: Dataframe containing all the informations and the right columns about the dataset
    :return: Augmented Dataframe containing the each hybrid component for each repository
    """
    resources_df = pd.DataFrame(
        columns=["repo", "labels", "dep_emb", "docstring_emb", "code_emb"]
    )
    print(len(dataset))
    dataset = dataset.dropna()
    j = 0
    print(len(dataset))
    for i, row in tqdm(dataset.iterrows()):
        print("embed index dataset:", i)
        repo_files = explore_repository(os.path.join(path, row["full_name"]))
        random.shuffle(repo_files)
        docstrings = []
        codes = []
        deps = []
        count_repo = 0
        dep_content = None
        # print("all relevant repo files:", repo_files)
        print("num_sampling_scripts: {}".format(num_sampling_scripts))
        while count_repo < num_sampling_scripts and len(repo_files) > 0:
            file = choose_file(repo_files, dep_content)
            file_content = read_script(file)
            if file_content is None:
                continue
            if pycg:
                dep_content = complete_dataset_pycg(file, path)
                print(file)
                print(dep_content)
                if len(dep_content) == 0:
                    continue
            else:
                dep_content = complete_dataset_ast(file_content)
                if len(dep_content[0]) == 0 and len(dep_content[1]) == 0:
                    continue
            docstring_content = get_docstring(file_content)
            docstrings.append(docstring_content)
            deps.append(dep_content)
            codes.append(file_content)
            count_repo += 1

        # Tokenizing all content for the repo
        code_input_tokens = [
            tokenizers["code"](
                file,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            for file in codes
        ]

        print("********************pycgContent*************************")
        print(deps)
        if pycg:
            dep_input_tokens = [
                tokenizers["english"](
                    "[SEP]".join(
                        [f'{" ".join(t[0].split(os.sep)[2:])} [C] {t[1]}' for t in ele]
                    ),
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                for ele in deps
            ]
        else:
            dep_input_tokens = [
                tokenizers["english"](
                    f"{t[0]} [SEP] {t[1]}",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                for t in deps
            ]
        print("********************doctrings*************************")
        print(docstrings)
        doc_input_tokens = [
            tokenizers["english"](
                doc,
                padding=True,
                add_special_tokens=True,
                truncation=True,
                return_tensors="pt",
            )
            for doc in docstrings
        ]
        emb_code = get_embeddings(code_input_tokens, "code")

        emb_dep = get_embeddings(dep_input_tokens, "english")

        emb_docstring = get_embeddings(doc_input_tokens, "english")
        if (
            len(emb_code) > 0
            and len(emb_code) == len(emb_dep)
            and len(emb_code) == len(emb_docstring)
        ):
            resources_df = resources_df.append(
                {
                    "repo": row["full_name"],
                    "labels": row["featured_topics"],
                    "dep_emb": emb_dep,
                    "docstring_emb": emb_docstring,
                    "code_emb": emb_code,
                },
                ignore_index=True,
            )
        else:
            pass
        if len(resources_df) > 50:
            resources_df.to_pickle(
                os.path.normpath(
                    os.path.join(ROOT_DIR, OUTPUT_PATH, f"resources_{j}_{num_sampling_scripts}.pkl")
                )
            )
            j += 1
            resources_df = pd.DataFrame(
                columns=["repo", "labels", "dep_emb", "docstring_emb", "code_emb"]
            )

    resources_df.to_pickle(os.path.normpath(os.path.join(path, f"resources_{j}_{num_sampling_scripts}.pkl")))
    return resources_df


def merge_resources(folder,num_sampling_scripts) -> None:
    """
    Function to merge all pickles containing the reposritories embeddings.

    :param folder: folder containing the resources pickles
    """
    filtered_files = [
        file for file in os.listdir(folder) if re.match(r"resources_\d+\.pkl", file)
    ]
    base_pickle = pd.read_pickle(os.path.join(folder, filtered_files[0]))
    for file in filtered_files[1:]:
        file_to_append = pd.read_pickle(os.path.join(folder, file))
        base_pickle = base_pickle.append(file_to_append, ignore_index=True)
    base_pickle.to_pickle(os.path.normpath(os.path.join(folder, 'resources_num_sampling_scripts_{}.pkl'.format(num_sampling_scripts))))


def get_resources(
    topics: List[str], type_embeddings: Union[List[str], str]
) -> pd.DataFrame:

    """
    Completes a given dataset by computing the desired embeddings on each of the repository

    :param topics: List of topics to keep the final dataset
    :param type_embeddings: List of type of embeddings to compose the hybrid script-level embeddings (Values: doctring, code, ast, pycg)
    :return: Augmented and topic-restricted dataset
    """

    infos = type_embeddings
    infos.extend(["repo", "labels"])
    dataset = pd.read_csv(
        os.path.normpath(os.path.join(ROOT_DIR, OUTPUT_PATH, RESOURCES_PKL))
    )
    dataset = dataset[infos].dropna()
    contains_topics = [
        True if len(list(set(topics).intersection(row["labels"]))) > 0 else False
        for i, row in dataset.iterrows()
    ]
    dataset = dataset[contains_topics]
    return dataset


def main():
    dataset_path = os.path.normpath(os.path.join(ROOT_DIR, DATASET_PATH, DATASET_CSV))
    os.environ["WANDB_DISABLED"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default=dataset_path, help="Path to dataset"
    )
    parser.add_argument("--num_sampling_scripts",type=int,default=15, help="Number of scripts sampled for repository embedding")

    args = parser.parse_args()

    print("Taking dataset from: ", args.path)

    dataset = pd.read_csv(os.path.join(args.path, "results_csv.csv"))
    resources = compute_resources(dataset, args.path,True,args.num_sampling_scripts)
    merge_resources(args.path,args.num_sampling_scripts)


if __name__ == "__main__":
    main()
