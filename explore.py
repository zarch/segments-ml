# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:36:42 2013

@author: pietro
"""
import random as rnd

import mlpy
import numpy as np
import pandas as pd

import plot as plt


def cls(ml, itraining, K_chk, y_chk, K_all=None):
    #import ipdb; ipdb.set_trace()
    ml.learn(np.array(K_chk[itraining].tolist()),
             np.array(y_chk[itraining].tolist()))
    K_all = K_all if K_all else K_chk
    return pd.Series(data=ml.pred(np.array(K_all.tolist())), index=K_all.index)


def test_gamma_C(x_arr, y_arr, itraining, K_chk, y_chk,
                 get_ml, get_kernel=None):
    shp = (len(y_arr), len(x_arr))
    pnl = pd.Panel(data=[np.zeros(shp), np.zeros(shp)],
                   items=['error', 'accuracy'],
                   major_axis=y_arr, minor_axis=x_arr)
    for g in x_arr:
        for c in y_arr:
            print "computing g=%r, c=%r" % (g, c)
            ml = get_ml(g, c, get_kernel)
            y_all = cls(ml, itraining, K_chk, y_chk)
            pnl['error'][g][c] = mlpy.error(y_chk, y_all[y_chk.index])
            pnl['accuracy'][g][c] = mlpy.accuracy(y_chk, y_all[y_chk.index])
    return pnl


def get_training_indexes(y, num):
    itran = []
    for cat in set(y):
        lst = list(y[y == cat].index)
        rnd.shuffle(lst)
        itran.extend(lst[:num])
    rnd.shuffle(itran)
    return pd.Index(itran)


def explore_domain(num, hdf, x_arr, y_arr, K_chk, y_chk,
                   get_ml, get_kernel=None):
    indexes = get_training_indexes(y_chk, num)
    strnum = 'explore_%d/%s' % (num, '%s') if num else 'explore_ALL/%s'
    indexes.to_series().to_hdf(hdf, strnum % 'indexes')
    pnl = test_gamma_C(x_arr, y_arr, indexes, K_chk, y_chk, get_ml, get_kernel)
    pnl.to_hdf(hdf, strnum % 'pnl')
    return pnl


def sensitivity(nums, hdf, x_arr, y_arr, K_chk, y_chk,
                get_ml, get_kernel=None, get_name=None, **plot):
    pnls = []
    for num in nums:
        pnl = explore_domain(num, hdf, x_arr, y_arr, K_chk, y_chk,
                             get_ml, get_kernel)
        if plot:
            plt.plot_df(pnl['accuracy'],
                        name = get_name(num) if get_name else '', **plot)
        pnls.append(pnl)
    return pnls

#
#exp_pnls = sensitivity(NUMS, HDF, x_arr, y_arr,
#                       get_kernel=lambda x: mlpy.KernelExponential(x),
#                       title='Accuracy, KernelExponential',
#                       subtitle='$best={a:.2f},\,\sigma={g:g},\,C={c:g}$',
#                       xlabel="$\sigma$", ylabel="$C$",
#                       name=DIR + "AccuracyExpDomain%r",
#                       typ='svg', rotation=90, fontsize=7)
#kexp = mlpy.KernelExponential(2)
#svm = mlpy.LibSvm(svm_type='c_svc', kernel=kexp, C=1)
#svm.learn(np.array(K_chk.tolist()), np.array(y_chk.tolist()))

