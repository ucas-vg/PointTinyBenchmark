"""
Code from paper
"A hybrid parameter estimation algorithm for beta mixtures
and applications to methylation state classification"
https://doi.org/10.1186/s13015-017-0112-1
https://bitbucket.org/genomeinformatics/betamix
"""

import numpy as np

from itertools import count
from argparse import ArgumentParser

import numpy as np
from scipy.stats import beta


def _get_values(x, left, right):
    y = x[np.logical_and(x>=left, x<=right)]
    n = len(y)
    if n == 0:
        m = (left+right) / 2.0
        v = (right-left) / 12.0
    else:
        m = np.mean(y)
        v = np.var(y)
        if v == 0.0:
            v = (right-left) / (12.0*(n+1))
    return m, v, n


def get_initialization(x, ncomponents, limit=0.8):
    # TODO: work with specific components instead of just their number
    points = np.linspace(0.0, 1.0, ncomponents+2)
    means = np.zeros(ncomponents)
    variances = np.zeros(ncomponents)
    pi = np.zeros(ncomponents)
    # init first component
    means[0], variances[0], pi[0] = _get_values(x, points[0], points[1])
    # init intermediate components
    N = ncomponents - 1
    for j in range(1, N):
        means[j], variances[j], pi[j] = _get_values(x, points[j], points[j+2])
    # init last component
    means[N], variances[N], pi[N] = _get_values(x, points[N+1], points[N+2])
    
    # compute parameters ab, pi
    ab = [ab_from_mv(m,v) for (m,v) in zip(means,variances)]
    pi = pi / pi.sum()
    
    # adjust first and last
    if ab[0][0] >= limit:  ab[0] = (limit, ab[0][1])
    if ab[-1][1] >= limit:  ab[-1] = (ab[-1][0], limit)
    return ab, pi


def ab_from_mv(m, v):
    """
    estimate beta parameters (a,b) from given mean and variance;
    return (a,b).

    Note, for uniform distribution on [0,1], (m,v)=(0.5,1/12)
    """
    phi = m*(1-m)/v - 1  # z = 2 for uniform distribution
    return (phi*m, phi*(1-m))  # a = b = 1 for uniform distribution


def get_weights(x, ab, pi):
    """return nsamples X ncomponents matrix with association weights"""
    bpdf = beta.pdf
    n, c = len(x), len(ab)
    y = np.zeros((n,c), dtype=float)
    s = np.zeros((n,1), dtype=float)
    for (j, p,(a,b)) in zip(count(), pi, ab):
        y[:,j] = p * bpdf(x, a, b)
    s = np.sum(y,1).reshape((n,1))
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        w = y / s  # this may produce inf or nan; this is o.k.!
    # clean up weights w, remove infs, nans, etc.
    wfirst = np.array([1] + [0]*(c-1), dtype=float)
    wlast = np.array([0]*(c-1) + [1], dtype=float)
    bad = (~np.isfinite(w)).any(axis=1)
    badfirst = np.logical_and(bad, x<0.5)
    badlast = np.logical_and(bad, x>=0.5)
    w[badfirst,:] = wfirst
    w[badlast,:] = wlast
    # now all weights are valid finite values and sum to 1 for each row
    assert np.all(np.isfinite(w)), (w, np.isfinite(w))
    assert np.allclose(np.sum(w,1), 1.0),  np.max(np.abs(np.sum(w,1)-1.0))
    return w


def relerror(x,y):
    if x==y:  return 0.0
    return abs(x-y)/max(abs(x),abs(y))

def get_delta(ab, abold, pi, piold):
    epi = max(relerror(p,po) for (p,po) in zip(pi,piold))
    ea =  max(relerror(a,ao) for (a,_), (ao,_) in zip(ab,abold))
    eb =  max(relerror(b,bo) for (_,b), (_,bo) in zip(ab,abold))
    return max(epi,ea,eb)


def estimate_mixture(x, init, steps=1000, tolerance=1E-5):
    """
    estimate a beta mixture model from the given data x
    with the given number of components and component types
    """
    (ab, pi) = init
    n, ncomponents = len(x), len(ab)

    for step in count():
        if step >= steps:  
            break
        abold = list(ab)
        piold = pi[:]
        # E-step: compute component memberships for each x
        w = get_weights(x, ab, pi)
        # compute component means and variances and parameters
        for j in range(ncomponents):
            wj = w[:,j]
            pij = np.sum(wj)
            m = np.dot(wj,x) / pij
            v = np.dot(wj,(x-m)**2) / pij
            if np.isnan(m) or np.isnan(v):
                m = 0.5;  v = 1/12  # uniform
                ab[j]=(1,1)  # uniform
                assert pij == 0.0
            else:
                assert np.isfinite(m) and np.isfinite(v), (j,m,v,pij)
                ab[j] = ab_from_mv(m,v)
            pi[j] = pij / n
        delta = get_delta(ab, abold, pi, piold)
        if delta < tolerance:
            break
    usedsteps = step + 1
    return (ab, pi, usedsteps)


def estimate(x, components, steps=1000, tolerance=1E-4):
    init = get_initialization(x, len(components))
    (ab, pi, usedsteps) = estimate_mixture(x, init, steps=steps, tolerance=tolerance)
    return (ab, pi, usedsteps)


class AccumHistogram1D():
    """https://raw.githubusercontent.com/NichtJens/numpy-accumulative-histograms/master/accuhist.py"""

    def __init__(self, nbins, xlow, xhigh):
        self.nbins = nbins
        self.xlow  = xlow
        self.xhigh = xhigh

        self.range = (xlow, xhigh)

        self.hist, edges = np.histogram([], bins=nbins, range=self.range)
        self.bins = (edges[:-1] + edges[1:]) / 2.

    def fill(self, arr):
        hist, _ = np.histogram(arr, bins=self.nbins, range=self.range)
        self.hist += hist

    @property
    def data(self):
        return self.bins, self.hist


