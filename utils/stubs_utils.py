"""
A module for caching and retrieving computational results to disk.

This module provides utility functions to save and load intermediate processing results,
which helps avoid redundant computations and speeds up development iterations.
"""

import os 
import pickle

# save a python object to a stub file
def save_stub(stub_path,object):
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))
        
    if stub_path is not None:
        with open(stub_path,'wb') as f:
            pickle.dump(object,f)

# read a python object from a stub file
def read_stub(read_from_stub,stub_path):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path,'rb') as f:
            object = pickle.load(f)
            return object
    return None
    