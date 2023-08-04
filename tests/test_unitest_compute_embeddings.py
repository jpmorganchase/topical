import sys
import os
from dotenv import load_dotenv
import unittest
import numpy as np

load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
TOPICAL_PATH = os.getenv('TOPICAL_PATH')
TEST_DIR = os.getenv('TEST_DIR')
sys.path.append(os.path.join(ROOT_DIR))

ENGLISH_MODEL = os.getenv('ENGLISH_MODEL')
CODE_MODEL = os.getenv('CODE_MODEL')
RESOURCES_PKL = os.getenv('RESOURCES_PKL')

from Topical.compute_embeddings import get_modules, get_first, complete_dataset_pycg, complete_dataset_ast, get_docstring, iter_json, choose_file, get_pycg, read_script, get_embeddings, get_tokens, explore_repository
from transformers import RobertaModel, RobertaTokenizer, DistilBertTokenizer, DistilBertModel

class ModuleTestCase(unittest.TestCase):
    def test_complete_dataset_pycg(self):
        file_input = os.path.join(ROOT_DIR, 'tests', 'dummy_repository', 'dummy_script.py')
        dep_input = os.path.join(ROOT_DIR, 'tests', 'dummy_repository')
        parse_pycg = complete_dataset_pycg(file_input, dep_input)
        output = [('dummy_script func_1', 'os path join'), ('dummy_script func_2', 'dummy_script2 func_2'), ('dummy_script2 func_2', 'os path exists'), ('dummy_script2 func_1', 'os path join')]
        assert parse_pycg == output

    def test_get_docstring(self):
        file_input = os.path.join(ROOT_DIR, 'tests', 'dummy_repository', 'dummy_script.py')
        file_content = read_script(file_input)
        docstring_content = get_docstring(file_content)
        docstring_output = "func 1 func 2 [SEP] This method is a test This method is also a test"
        assert docstring_content == docstring_output

    def test_get_embedding(self):
        ## INSTANCIATE BERT BASES AND THEIR ASSOCIATED TOKENIZERS
        tokenizers = {'code': RobertaTokenizer.from_pretrained(CODE_MODEL),
                      'english': DistilBertTokenizer.from_pretrained(ENGLISH_MODEL)
                      }
        models = {'code': RobertaModel.from_pretrained(CODE_MODEL),
                  'english': DistilBertModel.from_pretrained(ENGLISH_MODEL)}

        tokenizers['english'].add_special_tokens({'additional_special_tokens': ['[C]']})
        models['english'].resize_token_embeddings(len(tokenizers['english']))

        dep_content = None
        path = os.path.join(ROOT_DIR, 'tests', 'dummy_repository')
        repo_files = explore_repository(path)
        count_repo = 0
        docstrings = []
        codes = []
        deps = []
        while count_repo < 15 and len(repo_files) > 0:
            file = choose_file(repo_files, dep_content)
            file_content = read_script(file)
            if file_content is None:
                continue
            dep_content = complete_dataset_pycg(file, path)
            if len(dep_content) == 0:
                continue
            docstring_content = get_docstring(file_content)
            docstrings.append(docstring_content)
            deps.append(dep_content)
            codes.append(file_content)
            count_repo += 1

        # Tokenizing all content for the repo
        code_input_tokens = [tokenizers['code'](file,
                                                add_special_tokens=True,
                                                padding=True, truncation=True,
                                                return_tensors='pt') for file in codes]

        print('********************pycgContent*************************')
        print(deps, docstrings)

        dep_input_tokens = [
            tokenizers['english']("[SEP]".join([f'{" ".join(t[0].split(os.sep)[2:])} [C] {t[1]}' for t in ele]),
                                  add_special_tokens=True,
                                  padding=True,
                                  truncation=True,
                                  return_tensors='pt')
            for ele in deps]

        print('********************doctrings*************************')
        doc_input_tokens = [tokenizers['english'](doc, padding=True,
                                                  add_special_tokens=True,
                                                  truncation=True,
                                                  return_tensors='pt') for doc in docstrings]
        emb_code = get_embeddings(code_input_tokens, 'code')
        emb_dep = get_embeddings(dep_input_tokens, 'english')
        emb_docstring = get_embeddings(doc_input_tokens, 'english')
        checksum = np.sum(emb_code+emb_dep+emb_docstring)
        length = (len(doc_input_tokens[0]['input_ids'][0])+len(dep_input_tokens[0]['input_ids'][0])+len(code_input_tokens[0]['input_ids'][0]))
        assert abs(checksum-63.0) < 2.0
        assert length == 167


if __name__ == '__main__':
    unittest.main()