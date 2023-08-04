import os
import dummy_script2 as dummy2
"""
This script contains test methods
"""

def func_1(file):
    """
    This method is a test
    """
    os.path.join('.', file)
    return file

def func_2(file):
    """
    This method is also a test
    """
    file = dummy2.func_2(file)
    return file