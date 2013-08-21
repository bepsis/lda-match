#!/usr/bin/env python3
#
# Copyright (c) 2013 Kyle Gorman <gormanky@ohsu.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# This program allows the user to generate a subset of some sample
# (represented in with a comma-separated values file) in which two
# arbitrary groups differ non-significantly on an arbitrary number of
# continuous measures. The approach here is inspired by the "greedy"
# algorithm used by van Santen et al. (2010; Autism) but is likely to
# result in larger groups as it represents a close to "optimal" approach
# by using a linear discriminant analysis (LDA) projection of the
# continuous measures which are being matched. If -d is specified, 
# observations are only removed from A. If -d is not specified, then 
# observations are removed from the larger of the two residual groups, and
# from the first (-a) group in the case of a tie.
#
# This is the Python 3 version of this tool.

import numpy as np

from csv import DictReader, writer
from sys import argv, stdout, stderr
from getopt import getopt, GetoptError
from rpy2.robjects import packages, numpy2ri    # R in Python


## activate R functionality

stats = packages.importr('stats')               # R stats
numpy2ri.activate()                             # automatic numpy-to-R

## globals

DEFAULT_GROUP_COLUMN = ''
DEFAULT_OUTPUT_FID = stdout
DEFAULT_P_THRESHOLD = .2
MAX_DROP_PCT = .75
USAGE = """
USAGE: {} -a GRP1 -b GRP2 [-d] -m FEATS [-g GCOL] [-o OUTPUT] [-p P] INPUT
""".format(__file__)

## helper functions

def colmean(x):
    return np.mean(x, axis=0)

def colcov(x):
    return np.cov(x.T)

def rowsum(x):
    return np.sum(x, axis=1)

def read_csv(f, group_name, A_labels, B_labels):
    """
    Read a CSV file with column labels and return them and a list of
    fieldnames
    """
    source = DictReader(f)
    A_data = {feat: [] for feat in source.fieldnames}
    B_data = {feat: [] for feat in source.fieldnames}
    label = ''
    for row in source:
        label = row[group_name]
        if label in A_labels:
            for (key, value) in row.items():
                try:
                    value = float(value)
                except ValueError:
                    pass
                A_data[key].append(value)
        elif label in B_labels:
            for (key, value) in row.items():
                try:
                    value = float(value)
                except ValueError:
                    pass
                B_data[key].append(value)
    return (A_data, B_data, source.fieldnames)


def LDA(A, B):
    """
    Two-class LDA projection
    """
    # this isn't really defined in the case that A and B have only one
    # column, so we just return A and B, flipping them if mean(A) < mean(B)
    weights = 1.
    if A.shape[1] < 2:
        if colmean(A) < colmean(B):
            weights = -1.
    else:  # normal case
        icov = np.linalg.inv(colcov(A) + colcov(B))
        weights = np.dot(icov, colmean(A) - colmean(B))
    new_A = rowsum(A * weights)
    new_B = rowsum(B * weights)
    return (new_A, new_B)


def t_test(A, B, **Rkwargs):
    """
    Unequal variance two-sample t-test statistic and two-sided p-value
    """
    retval = stats.t_test(A, B, **Rkwargs)
    return (retval.rx2('statistic')[0], retval.rx2('p.value')[0])


def converged(A, B, p_thresh):
    """
    Perform independent sample t-test on each matched feature and return
    True iff all two-tailed p-values are >= p_thresh
    """
    for i in range(A.shape[1]):
        (_, p) = t_test(A[:, i], B[:, i])
        if p <= p_thresh:
            return False
    return True


def print_summary_statistics(A, B, matched_feats):
    """
    Prints summary statistics about the matched variables
    """
    print('Group output sizes: |A| = {}, |B| = {}'.format(
                                len(A),   len(B)), file=stderr)
    print('Matched feature summary statistics:', file=stderr)
    for (i, feat) in enumerate(matched_feats):
        (t, p) = t_test(A[:, i], B[:, i])
        print('\t{}:\tt = {: .2f}\tp(t) = {:.3f}'.format(feat, t, p), 
                                                        file=stderr)


def rotator(data, fieldnames):
    return zip(*(data[field] for field in fieldnames))


def write_csv(A_data, A_r, B_data, B_r, fieldnames, output_fid):
    """
    Prints resulting subset in CSV format
    """
    sink = writer(output_fid)
    sink.writerow(fieldnames)
    # constant time membership check
    Aset = set(A_r)
    Bset = set(B_r)
    # rotate data (to enable column iteration) while ignoring dropped
    # elements
    for (i, row) in enumerate(rotator(A_data, fieldnames)):
        if i in Aset:
            sink.writerow(row)
    for (i, row) in enumerate(rotator(B_data, fieldnames)):
        if i in Bset:
            sink.writerow(row)


if __name__ == '__main__':

    ## parse arguments
    try:
        (opts, args) = getopt(argv[1:], 'a:b:dg:hm:o:p:')
        # set default values
        GROUP_COLUMN = ''
        A_LABELS = set()
        B_LABELS = set()
        DROP_A_ONLY = False
        MATCHED_FEATS = set()
        OUTPUT_FID = DEFAULT_OUTPUT_FID
        P_THRESHOLD = DEFAULT_P_THRESHOLD
        # read flags
        for (opt, val) in opts:
            if opt == '-a':
                A_LABELS.update(val.split(','))
            elif opt == '-b':
                B_LABELS.update(val.split(','))
            elif opt == '-d':
                DROP_A_ONLY = True
            elif opt == '-g':
                GROUP_COLUMN = val
            elif opt == '-h':
                exit(USAGE)
            elif opt == '-m':
                MATCHED_FEATS.update(val.split(','))
            elif opt == '-o':
                try:
                    OUTPUT_FID = open(val, 'w')
                except IOError:
                    exit(USAGE + 'Error: cannot write to output (-o)')
            elif opt == '-p':
                try:
                    P_THRESHOLD = float(val)
                    if not (0. < P_THRESHOLD < 1.):
                        raise(ValueError)
                except ValueError:
                    exit(USAGE + 'Error: -p value must be > 0 and < 1.')
            else:
                raise(GetoptError)
    except GetoptError as err:
        exit(USAGE + 'Error: ' + str(err))
    if len(args) == 0:
        exit(USAGE + 'Error: No source file specified.')
    elif len(args) > 1:
        print(args)
        exit(USAGE + 'Error: More than one source file specified.')
    elif len(A_LABELS) == 0:
        exit(USAGE + 'Error: No first group (-a) specified.')
    elif len(B_LABELS) == 0:
        exit(USAGE + 'Error: No second group (-b) specified.')
    elif len(MATCHED_FEATS) == 0:
        exit(USAGE + 'Error: No matched features (-m) specified.')
    elif len(GROUP_COLUMN) == 0:
        exit(USAGE + 'Error: No or null group column (-g) specified.')
    elif len(A_LABELS & B_LABELS) > 0:  # intersection operation
        exit(USAGE + 'Error: First and second groups overlap.')

    ## read in source file
    print('Reading in source data...', end='', file=stderr)
    try:
        with open(args[0], 'rU') as f:
            (A_data, B_data, fieldnames) = read_csv(f, GROUP_COLUMN,
                                                    A_LABELS, B_LABELS)
    except (IOError, ValueError) as err:
        exit(USAGE + 'Error: {}'.format(str(err)))
    A_size = len(A_data[GROUP_COLUMN])
    B_size = len(B_data[GROUP_COLUMN])
    if A_size == 0:
        exit(USAGE + 'Error: No first group (-a) observations found.')
    elif B_size == 0:
        exit(USAGE + 'Error: No second group (-b) observations found.')
    print('done.', file=stderr)
    print('Group input sizes: |A| = {}, |B| = {}'.format(
        A_size, B_size), file=stderr)
    print('Projecting into LDA space...', end='', file=stderr)
    # check for missing features
    missing = MATCHED_FEATS - set(fieldnames)
    if len(missing) > 0:
        exit(USAGE + 'Error: Matched features "{}" not found'.format(
             ', '.join(missing)))
    # make matrices of A and B's matched features
    A_matrix = np.column_stack([A_data[feat] for feat in MATCHED_FEATS])
    B_matrix = np.column_stack([B_data[feat] for feat in MATCHED_FEATS])
    # project using LDA; projected A is greater on average than projected B
    try:
        (A_r, B_r) = LDA(A_matrix, B_matrix)
    except (ValueError, TypeError) as err:
        exit('\nError: Cannot match (-m) on non-numeric feature ' +
             '({})'.format(str(err)))
    ## overwrite projection with indices of the least extreme observations
    A_r = A_r.argsort()[::-1]  # more extreme values are highe
    B_r = B_r.argsort()       # more extreme values are lower
    print('done.', file=stderr)
    print('Removing subjects...', end='', file=stderr)
    # declare convergence failure after a fixed percentage of the
    # observations (MAX_DROP_PCT, by default 75%) have been removed
    MAX_DROP = int(MAX_DROP_PCT * (len(A_r) + len(B_r)))
    # split on -d
    Ai = 0
    if DROP_A_ONLY:
        # also stop if A becomes empty...
        for _ in range(min(MAX_DROP, len(A_r) - 1)):
            # check for convergence
            A_current = A_matrix[A_r[Ai:], :]
            if converged(A_current, B_matrix, P_THRESHOLD):
                print('done.', file=stderr)
                # show sample statistics
                print_summary_statistics(A_current, B_matrix,
                                         MATCHED_FEATS)
                write_csv(A_data, A_r[Ai:], B_data, B_r, fieldnames,
                                                         OUTPUT_FID)
                exit()
            # drop the most extreme point from the remainder of A
            Ai += 1
    else:
        Bi = 0
        for _ in range(MAX_DROP):
            # check for convergence from previous
            A_current = A_matrix[A_r[Ai:], :]
            B_current = B_matrix[B_r[Bi:], :]
            if converged(A_current, B_current, P_THRESHOLD):
                print('done.', file=stderr)
                # show sample statistics
                print_summary_statistics(A_current, B_current,
                                         MATCHED_FEATS)
                write_csv(A_data, A_r[Ai:], B_data, B_r[Bi:], fieldnames,
                                                              OUTPUT_FID)
                exit()
            # drop most extreme point from the larger of the remainder
            # of A and B on the next iteration
            if len(A_r) - Ai >= len(B_r) - Bi:
                Ai += 1
            else:
                Bi += 1
    # announce convergence failure if we get to this point
    exit('Error: Convergence failure ({} points removed)'.format(MAX_DROP))
