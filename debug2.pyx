import time
import numpy
cimport numpy

cdef unsigned long long int maxval
cdef unsigned long long int total
cdef int k
cdef double t1, t2, t
cdef numpy.ndarray arr

maxval = 10000
arr = numpy.arange(maxval)

t1 = time.time()

for k in arr:
    total = total + k
print "Total =", total

t2 = time.time()
t = t2 - t1
print("%.20f" % t)