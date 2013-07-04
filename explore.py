# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:36:42 2013

@author: pietro
"""
import random as rnd

import mlpy
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt


NUMS = [5, 10, 25, 50, 75, 100, None]
EXPLORE_START = -12
EXPLORE_STOP = 13
HDF = 'data.hdf'
KCHK = 'K_chk'
YCHK = 'y_chk'
KALL = 'df_stand'
IMGDIR = 'png/'
HDFDIR = 'hdf/'

def cls(svm, itraining, K_chk, y_chk, K_all=None):
    #import ipdb; ipdb.set_trace()
    svm.learn(np.array(K_chk[itraining].tolist()),
              np.array(y_chk[itraining].tolist()))
    K_all = K_all if K_all else K_chk
    return pd.Series(data=svm.pred(np.array(K_all.tolist())),
                     index=K_all.index)


def test_gamma_C(g_arr, c_arr, itraining, K_chk, y_chk, get_kernel):
    shp = (len(c_arr), len(g_arr))
    pnl = pd.Panel(data=[np.zeros(shp), np.zeros(shp)],
                   items=['error', 'accuracy'],
                   major_axis=c_arr, minor_axis=g_arr)
    for g in g_arr:
        for c in c_arr:
            print "computing g=%r, c=%r" % (g, c)
            svm = mlpy.LibSvm(svm_type='c_svc', kernel=get_kernel(g),
                              C=c, probability=True)
            y_all = cls(svm, itraining, K_chk, y_chk)
            pnl['error'][g][c] = mlpy.error(y_chk, y_all[y_chk.index])
            pnl['accuracy'][g][c] = mlpy.accuracy(y_chk, y_all[y_chk.index])
    return pnl


def get_training_indexes(y, num):
    itran = []
    for cat in set(y_chk):
        lst = list(y_chk[y_chk == cat].index)
        rnd.shuffle(lst)
        itran.extend(lst[:num])
    rnd.shuffle(itran)
    return pd.Index(itran)


def explore_domain(num, hdf, g_arr, c_arr, get_kernel):
    indexes = get_training_indexes(y_chk, num)
    strnum = 'explore_%d/%s' % (num, '%s') if num else 'explore_ALL/%s'
    indexes.to_series().to_hdf(hdf, strnum % 'indexes')
    pnl = test_gamma_C(g_arr, c_arr, indexes, K_chk, y_chk, get_kernel)
    pnl.to_hdf(hdf, strnum % 'pnl')
    return pnl


def plot_pnl(pnl, key='accuracy',
             title='',
             subtitle='$best={a:.2f},\,\gamma={g:g},\,C={c:g}$',
             xlabel="$\gamma$", ylabel="$C$",
             clr=plt.cm.Blues, rotation=45, fontsize=12,
             name='', typ='svg', dpi=600):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pclr = ax.pcolor(pnl[key], cmap=clr, edgecolors='0.5', linewidths=0.5)
    ax.set_xticks(np.arange(pnl.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(pnl.shape[2])+0.5, minor=False)
    #ax.xaxis.tick_top()
    ax.set_xticklabels(["%.0e" % g for g in pnl.minor_axis], minor=False,
                       rotation=rotation, fontsize=fontsize)
    ax.set_xbound(upper=pnl.shape[1])
    ax.set_xlabel(xlabel)
    ax.set_yticklabels(["%.0e" % c for c in pnl.major_axis], minor=False,
                       fontsize=fontsize)
    ax.set_xbound(upper=pnl.shape[2])
    ax.set_ylabel(ylabel)
    g, c = find_best_xy(pnl, key)
    sub = subtitle.format(a=pnl[key][g][c], g=g, c=c)
    ax.set_title("%s\n%s" % (title if title else key.title(), sub))
    ax.set_aspect(1)
    fig.colorbar(pclr, ax=ax, extend='both', spacing='uniform')  # shrink=0.9)
    if name:
        if isinstance(typ, str):
            plt.savefig('%s.%s' % (name, typ), dpi=dpi, format=typ,
                        transparent=True, bbox_inches='tight')
        else:
            for ty in typ:
                plt.savefig('%s.%s' % (name, typ), dpi=dpi, format=ty,
                            transparent=True, bbox_inches='tight')
    plt.show()


def find_best_xy(pnl, key):
    x_max = pnl[key].max()
    xi_max = x_max.idxmax()
    return xi_max, pnl[key][xi_max].idxmax()


#
# Read HDF
#
K_chk = pd.read_hdf(HDF, KCHK)
y_chk = pd.read_hdf(HDF, YCHK)
g_arr = 10.**np.arange(EXPLORE_START, EXPLORE_STOP)
c_arr = 10.**np.arange(EXPLORE_START, EXPLORE_STOP)


def sensitivity(nums, hdf, g_arr, c_arr, get_kernel,
                title='', subtitle='$best={a:.2f},\,\gamma={g:g},\,C={c:g}$',
                xlabel="$\gamma$", ylabel="$C$",
                name="AccuracyDomain%r",
                typ='svg', rotation=90, fontsize=7):
    pnls = []
    for num in nums:
        pnl = explore_domain(num, hdf, g_arr, c_arr, get_kernel)
        plot_pnl(pnl, key='accuracy',
                 title=title, subtitle=subtitle, xlabel=xlabel, ylabel=ylabel,
                 clr=plt.cm.PiYG,
                 name=name % num if num else "AccuracyDomainALL",
                 rotation=rotation, fontsize=fontsize, typ=typ)
        pnls.append(pnl)
    return pnls


#
# Support Vector Classification
#

def explore_SVM():
    linear_pnls = sensitivity(NUMS, HDFDIR + 'linear.hdf', np.array([1, ]), c_arr,
                              get_kernel=lambda x: mlpy.KernelLinear(),
                              title='Accuracy, KernelLinear',
                              subtitle='$best={a:.2f},\,C={c:g}$',
                              xlabel="$\sigma$", ylabel="$C$",
                              name=IMGDIR + "AccuracyLinearDomain%r",
                              typ='png', rotation=90, fontsize=7)

    poly_pnls = sensitivity(NUMS, HDFDIR + 'polynomial.hdf', g_arr, c_arr,
                            get_kernel=lambda x: mlpy.KernelPolynomial(x, b=1., d=2.),
                            title='Accuracy, KernelPolynomial',
                            subtitle='$best={a:.2f},\,\gamma={g:g},\,C={c:g}$',
                            xlabel="$\gamma$", ylabel="$C$",
                            name=IMGDIR + "AccuracyPolynomialDomainb1d2%r",
                            typ='png', rotation=90, fontsize=7)

    gauss_pnls = sensitivity(NUMS, HDFDIR + 'gaussian.hdf', g_arr, c_arr,
                             get_kernel=lambda x: mlpy.KernelGaussian(x),
                             title='Accuracy, KernelGaussian',
                             subtitle='$best={a:.2f},\,\sigma={g:g},\,C={c:g}$',
                             xlabel="$\sigma$", ylabel="$C$",
                             name=IMGDIR + "AccuracyGaussDomain%r",
                             typ='png', rotation=90, fontsize=7)
    #
    #exp_pnls = sensitivity(NUMS, HDF, g_arr, c_arr,
    #                       get_kernel=lambda x: mlpy.KernelExponential(x),
    #                       title='Accuracy, KernelExponential',
    #                       subtitle='$best={a:.2f},\,\sigma={g:g},\,C={c:g}$',
    #                       xlabel="$\sigma$", ylabel="$C$",
    #                       name=DIR + "AccuracyExpDomain%r",
    #                       typ='svg', rotation=90, fontsize=7)
    #kexp = mlpy.KernelExponential(2)
    #svm = mlpy.LibSvm(svm_type='c_svc', kernel=kexp, C=1)
    #svm.learn(np.array(K_chk.tolist()), np.array(y_chk.tolist()))

    sigm_pnls = sensitivity(NUMS, HDFDIR + 'sigmoid', g_arr, c_arr,
                            get_kernel=lambda x: mlpy.KernelSigmoid(x, b=1.),
                            title='Accuracy, KernelSigmoid',
                            subtitle='$best={a:.2f},\,\gamma={g:g},\,C={c:g}$',
                            xlabel="$\gamma$", ylabel="$C$",
                            name=IMGDIR + "AccuracySigmoidDomain%r",
                            typ='png', rotation=90, fontsize=7)

#
# Parzen-based classifier
#
#par = mlpy.Parzen(kernel=None)

#
# Kernel Fisher Discriminant Classifier
# lmb : float (>= 0.0) regularization parameter
#mlpy.KFDAC(lmb=0.001, kernel=None)

#
# k-Nearest-Neighbor
# k : int (>=1) number of nearest neighbors
#knn = mlpy.KNN(k)

#
# Classification Tree
# stumps : bool True: compute single split or False: standard tree
# minsize : int (>=0) minimum number of cases required to split a leaf
#tree = mlpy.ClassTree(stumps=0, minsize=1)

#
# Maximum Likelihood Classifier
#
#maxlkh = mlpy.MaximumLikelihoodC()
