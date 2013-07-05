# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:28:13 2013

@author: pietro
"""
import fnmatch
import re
import pandas as pd


def convert(text):
    return int(text) if text.isdigit() else text.lower()


def alphanum_key(key):
    return [convert(c) for c in re.split('([0-9]+)', key)]


def get_num(key):
    for e in alphanum_key(key):
        if not isinstance(e, str):
            return e


def get_df_one(hdfname):
    hdf = pd.HDFStore(hdfname)
    keys = sorted(fnmatch.filter(hdf.keys(), '*/pnl'), key=alphanum_key)
    data = dict()
    for key in keys:
        newkey = get_num(key)
        data[newkey if newkey else 'ALL'] = hdf[key]['accuracy'][1]
    cols = sorted(data.keys())
    return pd.DataFrame(data, index=data[cols[0]].index, columns=cols)
