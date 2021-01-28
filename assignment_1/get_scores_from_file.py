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

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(Entries, Id)
    ax.set_xlabel('Entry')
    ax.set_ylabel('Identity')
    ax.set_title('Mapping entry number to identity')
    pyplot.show()
    fig.savefig('Graphs/entry_number_id_entity_mapping.png')
    pyplot.close(fig)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(S)
    pyplot.show()
    fig.savefig('Graphs/scorematrix_data.png')
    pyplot.close(fig)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(gen_imp)
    pyplot.show()
    fig.savefig('Graphs/genuine_imposter_matrix.png')
    pyplot.close(fig)
