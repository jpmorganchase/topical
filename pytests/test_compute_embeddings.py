import sys
import os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
TOPICAL_PATH = os.getenv('TOPICAL_PATH')
TEST_DIR = os.getenv('TEST_DIR')
sys.path.append(os.path.join(ROOT_DIR, TOPICAL_PATH))

from compute_embeddings import get_modules, get_first, complete_dataset_pycg, complete_dataset_ast, get_docstring, iter_json, get_pycg, read_script, get_embeddings, get_tokens, explore_repository
import pytest

@pytest.mark.parametrize(
    "repo, paths",
    [
        (os.path.join(TEST_DIR, "test_repository"), [os.path.join(TEST_DIR,'test_repository\\test_script.py')])
    ]
)

def test_explore_repository(repo, paths):
    explored_paths = explore_repository(repo)
    assert explored_paths == paths

@pytest.mark.parametrize(
    "script_path, content",
    [
        (os.path.join(TEST_DIR, "test_repository\\test_script.py"), str)
    ]
)
def test_read_script(script_path, content):
    c = read_script(script_path)
    assert isinstance(c, content)

@pytest.mark.parametrize(
    "script_path, modules",
    [
        (os.path.join(TEST_DIR, "test_repository\\test_script.py"), ({'os'}, {'func_1', 'func_2'}))
    ]
)
def test_get_modules(script_path, modules):
    content = read_script(script_path)
    m = get_modules(content)
    assert m == modules

@pytest.mark.parametrize(
    "script_path, docstring",
    [
        (os.path.join(TEST_DIR, "test_repository\\test_script.py"),'func 1 func 2 [SEP] This method is a test This method is also a test')
    ]
)

def test_get_doctstring(script_path, docstring):
    content = read_script(script_path)
    d = get_docstring(content)
    assert d==docstring


@pytest.mark.parametrize(
    "script_path, output",
    [
        (os.path.join(TEST_DIR, "test_repository\\test_script.py"), {'os.path.exists': [], 'os.path.join': [], 'test\\test_repository\\test_script': [], 'test\\test_repository\\test_script.func_1': ['os.path.join'], 'test\\test_repository\\test_script.func_2': ['os.path.exists']})
    ]
)
def test_get_pycg(script_path, output):
    assert get_pycg(script_path) == output

@pytest.mark.parametrize(
    "pycg_output, deps",
    [
        ({os.path.join(TEST_DIR, 'test_repository\\test_script'): [],'dataset\\test_repository\\test_script.func_1': ['os.path.join'],'dataset\\test_repository\\test_script.func_2': ['os.path.exists'],'os.path.exists': [],'os.path.join': []},
         [('test_script func_1', 'os path join'), ('test_script func_2', 'os path exists')])
]
)

def test_iter_json(pycg_output, deps):
    parse_pycg = []
    iter_json(pycg_output, parse_pycg)
    assert parse_pycg == deps