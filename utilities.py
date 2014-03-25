#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by brown on 2010-07-28.
Copyright (c) 2010 Kevin S. Brown. All rights reserved.
"""

import numpy as np
from scipy import signal
from numpy import sort,int,floor,ceil,interp,std
from os import path

def empirical_ci(y,alpha=0.05):
	"""Computes an empirical (alpha/2,1-alpha/2) confidence interval for the distributional data in x.

	Parameters:
	------------
	x : numpy array, required
		set of data to produce empirical upper/lower bounds for

	alpha : float, optional
		sets desired CI range

	Returns:
	------------
	lb, ub : floats, lower and upper bounds for x

	"""
	ytilde = sort(y)
	xl = (alpha/2)*len(y)
	xu = (1.0 - alpha/2)*len(y)
	l1 = int(floor(xl))
	l2 = int(ceil(xl))
	u1 = int(floor(xu))
	u2 = int(ceil(xu))
	lb = interp(xl,[l1,l2],[ytilde[l1],ytilde[l2]])
	ub = interp(xu,[u1,u2],[ytilde[u1],ytilde[u2]])
	return lb,ub


def standardize(X,stdtype='column'):
	"""Standardizes a two dimensional input matrix X, by either row or column.  Resulting matrix will have 
	row or column mean 0 and row or column std. dev. equal to 1.0."""
	if len(X.shape) > 2:
		print 'ERROR: standardize() not defines for matrices that are not two-dimenional'
		return
	if stdtype == 'column':
		F = signal.detrend(X,type='constant',axis=0)
		F = F/std(F,axis=0)
	else:
	    F = signal.detrend(X.T,type='constant',axis=0)
	    F = F/std(F,axis=0)
	    F = F.T
	return F

def deconstruct_file_name(name):
    """Simple function to break apart a file name into individual pieces without the extension"""
    return path.splitext(path.split(name)[1])[0].split('_')

def construct_file_name(base,num,ext):
    """Simple function to create a file name of the form: base_num.ext"""
    return base+'_'+str(num)+'.'+ext 

def compute_unit_histogram(self,unitData,nBins=None):
	'''
	Computes a histogram of data bounded in [0,1] - for example absolute values of realization
	cross-correlation coefficients.'''
	rPDF = dict.fromkeys(['bin edges','counts','bar width'])
	if nBins is None:
		nBins = 101
	rPDF['bin edges'] = np.linspace(0,1.0,nBins)
	rPDF['counts'],_ = histogram(unitData,bins=rPDF['bin edges'])
	rPDF['bin edges'] = rPDF['bin edges'][0:-1]
	rPDF['bar width'] = 1.0/nBins
	return rPDF
