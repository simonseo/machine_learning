import numpy
import random

x = [[1,2,3],[4,5,6]]
y = [3,6]


x = numpy.array(x).T
x = numpy.append(x, [y], axis=0)
x = x.T.tolist()
random.shuffle(x)

print(x)
print(y)