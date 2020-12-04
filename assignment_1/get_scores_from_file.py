#!/usr/bin/env python

from matplotlib import pyplot
import numpy


def scores(id, S):
    [np, nt] = S.shape

    result = numpy.full(S.shape, False)

    for i in range(np):
        for j in range(i):
            if id[i] == id[j]:
                result[i, j] = True
    return result


if __name__ == "__main__":
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    [np, nt] = S.shape
    nId = Id.max()
    Entries = numpy.arange(1, np + 1)

    gen_imp = scores(Id, S)

    print('Size of score matrix {0} x {1}'.format(np, nt))
    print('Number of identities {0}'.format(nId))
    print('Number of genuine scores: {0}'.format(numpy.count_nonzero(gen_imp)))
    print('Number of impostor scores: {0}'.format((np * nt) - numpy.count_nonzero(gen_imp)))

    pyplot.plot(Entries, Id)
    pyplot.xlabel('Entry')
    pyplot.ylabel('Identity')
    pyplot.title('Mapping entry number to identity')
    pyplot.show()

    pyplot.imshow(S)
    pyplot.show()

    pyplot.imshow(gen_imp )
    pyplot.show()
