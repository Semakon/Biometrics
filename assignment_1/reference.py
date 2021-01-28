#!/usr/bin/env python

import numpy
import functions

if __name__ == "__main__":
    threshold = 0.5
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    filter = functions.filter_matrix(S)
    genuine_scores = functions.classify_scores(S, Id)
    imposter_scores = numpy.invert(genuine_scores) & filter
    thresholded_scores = functions.get_threshold_scores(filter, S, threshold)
    thresholded_genuine_scores = genuine_scores & thresholded_scores
    thresholded_imposter_scores = imposter_scores & thresholded_scores
    fmr = lambda t : functions.get_match_rate(filter, imposter_scores * S, t)
    tmr = lambda t : functions.get_match_rate(filter, genuine_scores * S, t)
    fnmr = lambda t : 1 - fmr(t)