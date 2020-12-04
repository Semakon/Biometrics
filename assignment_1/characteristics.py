#!/usr/bin/env python

from matplotlib import pyplot
import numpy
import functions

def plot_fmr(thresholds, fmr_values):
    pyplot.plot(thresholds, fmr_values)
    pyplot.xlabel('thresholds')
    pyplot.ylabel('fmr')
    pyplot.title('fmr')
    pyplot.show()

def plot_fnmr(thresholds, fnmr_values):
    pyplot.plot(thresholds, fnmr_values)
    pyplot.xlabel('thresholds')
    pyplot.ylabel('fnmr')
    pyplot.title('fnmr')
    pyplot.show()

def plot_mr(thresholds, fmr_values, fnmr_values):
    pyplot.plot(thresholds, fmr_values)
    pyplot.plot(thresholds, fnmr_values)
    pyplot.yscale('log')
    pyplot.xlabel('thresholds')
    pyplot.ylabel('mr')
    pyplot.title('mr')
    pyplot.show()

def plot_det(fmr_values, fnmr_values):
    pyplot.plot(fmr_values, fnmr_values)
    pyplot.plot(fmr_values, fmr_values, '--')
    pyplot.xscale('log')
    pyplot.xlabel('fmr')
    pyplot.ylabel('fnmr')
    pyplot.title('det')
    pyplot.show()

def plot_roc(fmr_values, tmr_values):
    pyplot.plot(fmr_values, tmr_values)
    pyplot.xlabel('fmr')
    pyplot.ylabel('tmr')
    pyplot.title('roc')
    pyplot.show()

if __name__ == "__main__":
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    genuine_scores = functions.classify_scores(S, Id)
    imposter_scores = numpy.invert(genuine_scores)
    fmr = lambda t : functions.get_match_rate(imposter_scores * S, t)
    tmr = lambda t : functions.get_match_rate(genuine_scores * S, t)
    fnmr = lambda t : 1 - tmr(t)
    thresholds = range(0, 800)
    fmr_values = numpy.asarray([fmr(i) for i in thresholds])
    tmr_values = numpy.asarray([tmr(i) for i in thresholds])
    fnmr_values = 1 - tmr_values

    plot_fmr(thresholds, fmr_values)
    plot_fnmr(thresholds, fnmr_values)
    plot_mr(thresholds, fmr_values, fnmr_values)
    plot_det(fmr_values, fnmr_values)
    plot_roc(fmr_values, tmr_values)
