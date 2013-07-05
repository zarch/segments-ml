# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:45:55 2013

@author: pietro
"""
import explore as ex
import explore_config as ec


def get_conf(keys, conf, update=None):
    """Return a dictionary inherit from other dictionary in conf. ::

        >>> conf = dict(a={'a': 0, 'b': 1, 'c': 2},
        ...             b={'inh': 'a', 'b': 10},
        ...             c={'inh': 'b', 'c': 20, 'd': 30})
        >>> get_conf('c', conf)
        {'a': 0, 'c': 20, 'b': 10, 'd': 30}

    """
    keys = [keys, ] if isinstance(keys, str) else keys
    dic = dict() if update is None else update
    for key in keys:
        ikeys = conf[key].keys()
        if 'inh' in ikeys:
            dic = get_conf(conf[key]['inh'], conf, update=dic)
            ikeys.remove('inh')
        for k in ikeys:
            dic[k] = conf[key][k]
    return dic


def run_conf(conf, target=None):
    for key in conf:
        print key
        kwargs = get_conf(key, conf)
        if kwargs.pop('run'):
            if target is None:
                target = kwargs.pop('taget')
            target(**kwargs)


run_conf(ec.SVM_TESTS, target=ex.sensitivity)
