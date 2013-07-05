# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:28:13 2013

@author: pietro
"""
import numpy as np
import pandas as pd
import mlpy

NUMS = [5, ]  #10]  #, 25, 50, 75, 100, None]
EXPLORE_START = -12
EXPLORE_STOP = 13
HDF = 'data.hdf'
KCHK = 'K_chk'
YCHK = 'y_chk'
KALL = 'df_stand'
IMGDIR = 'pngt/'
HDFDIR = 'hdft/'

x_arr = 10.**np.arange(EXPLORE_START, EXPLORE_STOP)
y_arr = 10.**np.arange(EXPLORE_START, EXPLORE_STOP)

#
# Read HDF
#
K_chk = pd.read_hdf(HDF, KCHK)
y_chk = pd.read_hdf(HDF, YCHK)


#
# Parzen-based classifier
#
#par = mlpy.Parzen(kernel=None)

def get_parzen(x, y, get_kernel=None):
    return mlpy.Parzen(kernel=get_kernel(x) if get_kernel else None)

PARZEN_TESTS = {
}


#
# Kernel Fisher Discriminant Classifier
# lmb : float (>= 0.0) regularization parameter
#mlpy.KFDAC(lmb=0.001, kernel=None)

def get_kfdac(x, y, get_kernel=None):
    return mlpy.KFDAC(lmb=y, kernel=get_kernel(x) if get_kernel else None)

KFDAC_TESTS = {
}


#
# k-Nearest-Neighbor
# k : int (>=1) number of nearest neighbors
#knn = mlpy.KNN(k)

def get_knn(x, y, get_kernel=None):
    return mlpy.KNN(x)


#
# Classification Tree
# stumps : bool True: compute single split or False: standard tree
# minsize : int (>=0) minimum number of cases required to split a leaf
#tree = mlpy.ClassTree(stumps=0, minsize=1)

def get_tree(x, y, get_kernel=None):
    return mlpy.ClassTree(stumps=x, minsize=y)


#
# Maximum Likelihood Classifier
#
#maxlkh = mlpy.MaximumLikelihoodC()


#
# Support Vector Classification
#

def get_svm(x, y, get_kernel=None):
    try:
        return mlpy.LibSvm(svm_type='c_svc',
                       kernel=get_kernel(x) if get_kernel else None, C=y,
                       probability=True)
    except ValueError:
        import ipdb; ipdb.set_trace()

KNN_TESTS = {
    "KNN": {'nums': NUMS,
               'x_arr': np.array([1, ]),
               'y_arr': y_arr,
               'get_ml': get_parzen,
               'subtitle': '$best={a:.2f},\,\gamma={c:g}$',
               'xlabel': "$\gamma$",
               'ylabel': "$C$",
               'typ': 'png',
               'rotation': 90,
               'fontsize': 7,
               'run': False, },
}


TREE_TESTS = {
}


def get_name(label, func=repr):
    def get(x):
        return label % func(x) if x else label % 'all'
    return get

SVM_TESTS = {
    "SVM": {'nums': NUMS,
            'x_arr': x_arr,
            'y_arr': y_arr,
            'K_chk': K_chk,
            'y_chk': y_chk,
            'get_ml': get_svm,
            'title': 'Accuracy',
            'subtitle': '$best={a:.2f},\,\gamma={g:g},\,C={c:g}$',
            'xlabel': "$\gamma$",
            'ylabel': "$C$",
            'typ': 'png',
            'rotation': 90,
            'fontsize': 7,
            'run': False, },
    "linear": {'inh': "SVM",
               'hdf': HDFDIR + 'linear.hdf',
               'x_arr': np.array([1, ]),
               'get_kernel': lambda x: mlpy.KernelLinear(),
               'title': 'Accuracy, KernelLinear',
               'subtitle': '$best={a:.2f},\,C={c:g}$',
               'xlabel': "$\sigma$",
               'get_name': get_name(IMGDIR + "AccuracyLinearDomain%s"),
               'run': False},
    "polygon": {'inh': "SVM",
                'hdf': HDFDIR + 'polynomial.hdf',
                'get_name': get_name(IMGDIR + "AccuracyPolygonDomain%s"),
                'get_kernel': lambda x: mlpy.KernelPolynomial(x, b=1., d=2.),
                'title': 'Accuracy, KernelPolynomial',
                'run': True, },
    "gauss": {'inh': "linear",
              'hdf': HDFDIR + 'gaussian.hdf',
              'x_arr': x_arr,
              'y_arr': y_arr,
              'get_kernel': lambda x: mlpy.KernelGaussian(x),
              'title': 'Accuracy, KernelGaussian',
              'subtitle': '$best={a:.2f},\,\sigma={g:g},\,C={c:g}$',
              'get_name': get_name(IMGDIR + "AccuracyGaussDomain%s"),
              'run': True},
    "sigmoid": {'inh': "SVM",
                'hdf': HDFDIR + 'sigmoid.hdf',
                'get_kernel': lambda x: mlpy.KernelSigmoid(x, b=1.),
                'title': 'Accuracy, KernelSigmoid',
                'get_name': get_name(IMGDIR + "AccuracySigmoidDomain%s"),
                'run': True},
}
