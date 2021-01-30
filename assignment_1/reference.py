#!/usr/bin/env python

import numpy
import functions

if __name__ == "__main__":
    threshold = 0.5
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    filter = functions.filter_matrix(S)
    min_val = int(numpy.floor(numpy.min(filter * S)))
    max_val = int(numpy.ceil(numpy.max(filter * S)))
    qualifier = min_val - 1
    filter_val = min_val - max_val - 2

    if not greater:
        qualifier = max_val + 1
        filter_val = -filter_val

    classified_scores = functions.classify_scores(S, Id)
    genuine_scores = functions.filter_scores(S, classified_scores, filter_val)
    imposter_scores = functions.filter_scores(S, numpy.invert(classified_scores) & filter, filter_val)
    thresholded_scores = functions.get_threshold_scores(S, t, greater = True)
    thresholded_genuine_scores = functions.get_threshold_scores(genuine_scores, t, greater)
    thresholded_imposter_scores = functions.get_threshold_scores(imposter_scores, t, greater)
    fmr = lambda t : functions.get_match_rate(imposter_scores, qualifier, t, greater)
    tmr = lambda t : functions.get_match_rate(genuine_scores, qualifier, t, greater)
    fnmr = lambda t : 1 - fmr(t)
