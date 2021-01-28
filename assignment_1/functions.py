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

def get_match_rate(filter, scores, t, greater = True):
    thresholded_scores = get_threshold_scores(filter, scores, t, greater)
    thresholded_scores_count = numpy.count_nonzero(thresholded_scores)
    scores_count = numpy.count_nonzero(scores)
    return thresholded_scores_count / scores_count

def get_threshold_scores(filter, S, t, greater = True):
    result = numpy.greater(S, t)

    if not greater:
        result = numpy.invert(result)

    return result & filter
