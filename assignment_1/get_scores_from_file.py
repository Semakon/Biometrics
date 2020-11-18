from matplotlib import pyplot
import numpy

if __name__ == "__main__":
    Id = numpy.loadtxt('id.txt')
    S = numpy.loadtxt('scorematrix.txt')
    [np, nt] = S.shape
    nId = Id.max()
    Entries = numpy.arange(1, np + 1)

    print('Size of score matrix {0} x {1}'.format(np, nt))
    print('Number of identities {0}'.format(nId))

    pyplot.plot(Entries, Id)
    pyplot.xlabel('Entry')
    pyplot.ylabel('Identity')
    pyplot.title('Mapping entry number to identity')
    pyplot.show()

    pyplot.imshow(S)
    pyplot.show()
