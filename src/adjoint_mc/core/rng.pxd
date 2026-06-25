# cython: language_level=3

cdef struct Pcg64State:
    unsigned long long state
    unsigned long long inc

cdef void pcg64_seed(Pcg64State* st, unsigned long long seed) nogil noexcept

cdef void pcg64_advance(Pcg64State* st, unsigned long long seed, unsigned long long inc) nogil noexcept

cdef double pcg64_random(Pcg64State* st) nogil noexcept

cdef double pcg64_normal(Pcg64State* st) nogil noexcept
