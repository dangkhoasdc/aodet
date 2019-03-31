"""
Helper functions
"""
import os
import warnings
import yaml
import hashlib
import numpy as np


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()

def intersect_list(l1, l2):
    ll1 = l1 if l1 is not None else []
    ll2 = l2 if l2 is not None else []
    return list(set(ll1).intersection(set(ll2)))


def _to_ints(*args):
    """convert a list of other types to int"""
    return list(map(int, args))


def _to_floats(*args):
    """convert a list of other types to int"""
    return list(map(float, args))

def concate_dict(*args):
    """ concatenate dicts """
    dall = {}
    for d in args:
        if d is None:
            continue
        dall.update(d)

    return dall

def index(l, func):
    """
    find the index which is satisfy the func
    Argument:
        :param l: an input list
        :param func: the function returns boolean value
    Returns:
        :return: return the index otherwise raise Exception
    """

    for idx, item in enumerate(l):
        if func(item):
            return idx

    raise ValueError("Could not find the item")

def contain_substr(s, prefixes):
    """
    check if whether any substring in a set is contained in a string.
    Parameters:
        :param s: input string
        :peram prefixes: a set of substrings
    """

    for p in prefixes:
        if p in s:
            return True

    return False

def default_zeros(shape, dtype=float, order='C'):
    """
    create a zero array for defautdict
    Usage:
        x = defaultdict(default_zeros(shape))
    Paramters:
        :param shape: shapeension of the array
    """
    def _create_zeros_array():
        return np.zeros(shape, dtype=dtype, order=order)

    return _create_zeros_array


def comp_list(l1, l2):
    return len(l1) == len(l2) and len(set(l1).intersection(l2)) == len(l1)


def f1(p, r):
    if p == 0.0 and r ==0.0:
        return 'NaN'
    return 2.0*(p*r)/(p+r)
