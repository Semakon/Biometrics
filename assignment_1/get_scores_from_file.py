from matplotlib import pyplot
import numpy
from os import path


def scores(id, S):
    [np, nt] = S.shape

    result = numpy.zeros(S.shape)

    for i in range(np):
        for j in range(i):
            if id[i] == id[j]:
                result[i, j] = 1
    return result


def fmr(imp, t):
    imp_over_t = 0
    for s in imp:
        if s > t:
            imp_over_t += 1
    return imp_over_t / len(imp)


def fnmr(gen, t):
    gen_over_t = 0
    for s in gen:
        if s > t:
            gen_over_t += 1
    return 1 - (gen_over_t / len(gen))


def calc_fmr_fnmr(Id, S):
    # Find genuine and impostor scores
    gen_imp = scores(Id, S)

    # Make genuine and impostor lists
    gen = []
    imp = []
    for i in range(len(gen_imp)):
        for j in range(len(gen_imp[i])):
            if gen_imp[i, j] == 0:
                imp.append(S[i, j])
            else:
                gen.append(S[i, j])
    # Determine min and max threshold
    min_s = numpy.min(S)
    max_s = numpy.max(S)

    # print('Lowest Score: {0}'.format(min_s))
    # print('Highest Score: {0}'.format(max_s))
    # print('Average genuine score: {0}'.format(numpy.mean(gen)))
    # print('Average impostor score: {0}'.format(numpy.mean(imp)))
    # print()

    # Determine FMR as a function of decision threshold
    thresholds = numpy.arange(int(min_s), int(max_s), 1)
    fmr_t = numpy.zeros(len(thresholds))
    for i in range(len(thresholds)):
        fmr_t[i] = fmr(imp, thresholds[i])

    # Determine FNMR as function of decision threshold
    fnmr_t = numpy.zeros(len(thresholds))
    for i in range(len(thresholds)):
        fnmr_t[i] = fnmr(gen, thresholds[i])

    # Save thresholds, FMR, and FNMR to file
    numpy.savetxt('thresholds.txt', thresholds)
    numpy.savetxt('fmr_t.txt', fmr_t)
    numpy.savetxt('fnmr_t.txt', fnmr_t)

    return thresholds, fmr_t, fnmr_t


def show_graphs(thresholds, fmr_t, fnmr_t):
    # Plot FMR and FNMR on same graph
    pyplot.plot(thresholds, fmr_t, label='FMR')
    pyplot.plot(thresholds, fnmr_t, label='FNMR')
    pyplot.xlabel('Threshold')
    pyplot.title('FMR and FNMR as a function of decision threshold')
    pyplot.legend()
    pyplot.show()

    # Plot DET curve
    pyplot.plot(fmr_t, fnmr_t)
    pyplot.plot([0, 1], [0, 1], 'b--')
    pyplot.xlabel('FMR(t)')
    pyplot.ylabel('FNMR(t)')
    pyplot.title('Decision Error Trade-off (DET) curve')
    pyplot.show()

    # Plot ROC curve
    pyplot.plot(fmr_t, 1 - fnmr_t)
    pyplot.xlabel('FMR(t)')
    pyplot.ylabel('TMR(t)')
    pyplot.title('Receiver Operating Characteristic (ROC) curve')
    pyplot.show()


if __name__ == "__main__":
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    [np, nt] = S.shape
    nId = numpy.max(Id)
    Entries = numpy.arange(1, np + 1)

    # print('Size of score matrix: {0} x {1}'.format(np, nt))
    # print('Number of identities: {0}'.format(nId))
    # print('Number of genuine scores: {0}'.format(len(gen)))
    # print('Number of impostor scores: {0}'.format(len(imp)))
    # print()

    # Get FMR(t), FNMR(t), and the thresholds
    if path.exists('thresholds.txt') and path.exists('fmr_t.txt') and path.exists('fnmr_t.txt'):
        # Get FMR(t) and FNMR(t) from file
        thresholds = numpy.loadtxt('thresholds.txt')
        fmr_t = numpy.loadtxt('fmr_t.txt')
        fnmr_t = numpy.loadtxt('fnmr_t.txt')
    else:
        # Calculate FMR(t) and FNMR(t)
        thresholds, fmr_t, fnmr_t = calc_fmr_fnmr(Id, S)

    # Plot the FMR, FNMR, DET curve, and ROC curve
    show_graphs(thresholds, fmr_t, fnmr_t)
