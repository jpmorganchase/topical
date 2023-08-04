import sys
import os
from dotenv import load_dotenv
import unittest

load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
TOPICAL_PATH = os.getenv('TOPICAL_PATH')
TEST_DIR = os.getenv('TEST_DIR')
sys.path.append(os.path.join(ROOT_DIR, TOPICAL_PATH))

from compute_embeddings import get_modules, get_first, complete_dataset_pycg, complete_dataset_ast, get_docstring, iter_json, get_pycg, read_script, get_embeddings, get_tokens, explore_repository
import pytest

class ModuleTestCase(unittest.TestCase):
    def test_iter_json(self):
        parse_pycg = []
        pycg_output = []
        deps = []
        iter_json(pycg_output, parse_pycg)
        print(parse_pycg)
        assert parse_pycg == deps