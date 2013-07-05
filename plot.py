# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:28:13 2013

@author: pietro
"""
import numpy as np
import matplotlib.pyplot as plt


def find_best_xy(df):
    x_max = df.max()
    xi_max = x_max.idxmax()
    return xi_max, df[xi_max].idxmax()


def get_coords(xindex, yindex, df, s=0):
    if hasattr(xindex, '__iter__'):
        x = [df.columns.get_loc(xi) + s for xi in xindex]
        y = [df.index.get_loc(yi) + s for yi in yindex]
        return x, y
    return df.columns.get_loc(xindex) + s, df.index.get_loc(yindex) + s


def plot_df(df, title='Accurary',
            subtitle='$best={a:.2f},\,\gamma={g:g},\,C={c:g}$',
            xlabel="$\gamma$", ylabel="$C$",
            xticklabels=lambda x: "%.0e" % x,
            yticklabels=lambda x: "%.0e" % x,
            aspect=1,
            clr=plt.cm.Blues, rotation=45, fontsize=12,
            name='', typ='svg', dpi=600, best='r.', onebest=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pclr = ax.pcolor(df, cmap=clr, edgecolors='0.5', linewidths=0.5)
    ax.set_yticks(np.arange(df.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(df.shape[1])+0.5, minor=False)
    #ax.xaxis.tick_top()
    ax.set_xticklabels([xticklabels(g) for g in df.columns], minor=False,
                       rotation=rotation, fontsize=fontsize)
    ax.set_xbound(upper=df.shape[0])
    ax.set_xlabel(xlabel)
    #import ipdb; ipdb.set_trace()
    ax.set_yticklabels([yticklabels(c) for c in df.index], minor=False,
                       fontsize=fontsize)
    ax.set_xbound(upper=df.shape[1])
    ax.set_ylabel(ylabel)
    g, c = find_best_xy(df)
    sub = subtitle.format(a=df[g][c], g=g, c=c)
    ax.set_title("%s\n%s" % (title, sub))
    ax.set_aspect(aspect)
    fig.colorbar(pclr, ax=ax, extend='both', spacing='uniform')  # shrink=0.9)
    if best:
        if onebest:
            ximax, yimax = find_best_xy(df)
            x, y = get_coords(ximax, yimax, df, 0.5)
        else:
            dm = df.idxmax()
            ximax, yimax = dm.index, dm.values
            x, y = get_coords(ximax, yimax, df, 0.5)
        ax.plot(x, y, best)
        #ax.plot(np.arange(len(df.columns)) + 0.5, df.idxmax(), best)
    if name:
        if isinstance(typ, str):
            plt.savefig('%s.%s' % (name, typ), dpi=dpi, format=typ,
                        transparent=True, bbox_inches='tight')
        else:
            for ty in typ:
                plt.savefig('%s.%s' % (name, typ), dpi=dpi, format=ty,
                            transparent=True, bbox_inches='tight')
    plt.show()




#df = get_df_one('hdf/linear.hdf')
#
#plot_df(df, title="Accuracy, KernelLinear",
#        subtitle="",
#        xlabel="Number of trining areas",
#        ylabel="C",
#        xticklabels=lambda x: str(x),
#        aspect=1, rotation=90, fontsize=7, best='r.')
