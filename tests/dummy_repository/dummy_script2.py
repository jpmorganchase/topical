import os
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
    return os.path.exists(file)