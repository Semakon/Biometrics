#!/usr/bin/env python

from matplotlib import pyplot
import numpy
import functions

def plot_fmr(thresholds, fmr_values):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, fmr_values)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('fmr')
    ax.set_title('fmr')
    pyplot.show()
    fig.savefig('Graphs/fmr.png')
    pyplot.close(fig)

def plot_fnmr(thresholds, fnmr_values):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, fnmr_values)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('fnmr')
    ax.set_title('fnmr')
    pyplot.show()
    fig.savefig('Graphs/fnmr.png')
    pyplot.close(fig)

def plot_mr(thresholds, fmr_values, fnmr_values):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, fmr_values)
    ax.plot(thresholds, fnmr_values)
    # ax.set_yscale('log')
    ax.set_xlabel('thresholds')
    ax.set_ylabel('mr')
    ax.set_title('mr')
    pyplot.show()
    fig.savefig('Graphs/mr.png')
    pyplot.close(fig)

def plot_det(fmr_values, fnmr_values):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(fmr_values, fnmr_values)
    ax.plot(fmr_values, fmr_values, '--')
    # ax.set_xscale('log')
    ax.set_xlabel('fmr')
    ax.set_ylabel('fnmr')
    ax.set_title('det')
    pyplot.show()
    fig.savefig('Graphs/det.png')
    pyplot.close(fig)

def plot_roc(fmr_values, tmr_values):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(fmr_values, tmr_values)
    ax.set_xlabel('fmr')
    ax.set_ylabel('tmr')
    ax.set_title('roc')
    pyplot.show()
    fig.savefig('Graphs/roc.png')
    pyplot.close(fig)

if __name__ == "__main__":
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    filter = functions.filter_matrix(S)
    genuine_scores = functions.classify_scores(S, Id)
    imposter_scores = numpy.invert(genuine_scores) & filter
    fmr = lambda t : functions.get_match_rate(filter, imposter_scores * S, t)
    tmr = lambda t : functions.get_match_rate(filter, genuine_scores * S, t)
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
