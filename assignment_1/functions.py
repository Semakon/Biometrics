#!/usr/bin/env python

import numpy

def classify_scores(S, id):
    [np, _] = S.shape
    result = numpy.full(S.shape, False)

    for i in range(np):
        for j in range(i):
            if id[i] == id[j]:
                result[i, j] = True

    return result

def filter_matrix(S):
    [np, _] = S.shape
    result = numpy.full(S.shape, False)

    for i in range(np):
        for j in range(i):
            result[i, j] = True

    return result

def filter_scores(S, filter, filter_value):
    filter = numpy.invert(filter) * 1
    filter[filter == 1] = filter_value
    return S + filter

def get_match_rate(scores, qualifier, t, greater = True):
    thresholded_scores = get_threshold_scores(scores, t, greater)
    thresholded_scores_count = numpy.count_nonzero(thresholded_scores)
    qualified_scores = get_threshold_scores(scores, qualifier, greater)
    scores_count = numpy.count_nonzero(qualified_scores)
    return thresholded_scores_count / scores_count

def get_threshold_scores(S, t, greater = True):
    result = numpy.greater(S, t)

    if not greater:
        result = numpy.invert(result)

    return result
