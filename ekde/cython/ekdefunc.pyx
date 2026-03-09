# distutils: language = c++

from libcpp cimport bool
from libc.stdlib cimport malloc, free
import cython

from libc.math cimport exp
from libc.math cimport pow as cpow

from tqdm import tqdm

import numpy as np
cimport numpy as np

from cpython cimport array
import array

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void count_diff_desc(int[:,:] A,
               int[:,:] out):
    cdef Py_ssize_t j, i_asc, i

    cdef int n = A.shape[0]
    cdef int d = A.shape[1]

    for i_asc in range(n - 1):
        i = n - 2 - i_asc
        for j in range(d):
            if A[i,j] == A[i+1, j]:
                if j == 0:
                    out[i,j] = out[i+1,j] + 1
                elif out[i,j-1] > 1:
                    out[i,j] = out[i+1,j] + 1

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void count_diff_asc(int[:,:] A,
                          int[:,:] out):
    cdef Py_ssize_t j, i

    cdef int n = A.shape[0]
    cdef int d = A.shape[1]

    for i in range(1, n):
        for j in range(d):
            if A[i,j] == A[i-1, j]:
                if j == 0:
                    out[i,j] = out[i-1,j] + 1
                elif out[i,j-1] > 1:
                    out[i,j] = out[i-1,j] + 1


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int *** sparse(int[:,:] A,
                    int[:,:] A_diff_desc,
                    int[:,:] A_diff_one_side,
                    int *S_shape):
    cdef Py_ssize_t i_A, i_S, j, j_asc

    cdef int n = A.shape[0]
    cdef int d = A.shape[1]

    cdef int ***S = <int ***> malloc(d *sizeof(int **))

    for j in range(d):
        S[j] = <int **> malloc(S_shape[j] * sizeof(int *))

    for i_A in range(n):
        S[d-1][i_A] = <int *> malloc(2 * sizeof(int))
        S[d-1][i_A][0] = A[i_A, d-1]
        S[d-1][i_A][1] = i_A

    for j in range(d-1):
        S[j][0] = <int *> malloc(2 * sizeof(int))
        S[j][0][0] = A[0, j]
        S[j][0][1] = 0

    for j in range(d-1):
        i_A = 0
        for i_S in range(1, S_shape[j]):
            S[j][i_S] = <int *> malloc(2 * sizeof(int))

            S[j][i_S][1] = S[j][i_S - 1][1] + A_diff_one_side[i_A, j]

            i_A = i_A + A_diff_desc[i_A, j]

            S[j][i_S][0] = A[i_A, j]

    return(S)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int [:,:] count_one_side(int[:,:] A_diff_desc):
    cdef Py_ssize_t i, j, i_asc

    cdef int n = A_diff_desc.shape[0]
    cdef int d = A_diff_desc.shape[1]
    cdef int [:,:] A_diff_one_side = np.zeros((n, d-1), dtype=np.intc)

    cdef int cnt

    for j in range(d-1):
        cnt = 0
        for i_asc in range(n):
            i = n - 1 - i_asc
            if A_diff_desc[i, j] == 1:
                cnt = 0

            if A_diff_desc[i, j+1] == 1:
                cnt = cnt + 1

            A_diff_one_side[i, j] = cnt

    return(A_diff_one_side)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int exponential_search_left(int **L,
                                 int x,
                                 int a,
                                 int b):
    cdef int bound = 1
    cdef size = b - a

    while bound < size and L[a + bound][0] < x:
        bound = bound * 2

    return(binary_search_left(L=L, x=x, a=a + bound//2, b=min(a + bound + 1, b)))

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int exponential_search_right(int **L,
                                  int x,
                                  int a,
                                  int b):
    cdef int bound = 1
    cdef size = b - a

    while bound < size and L[a + bound][0] <= x:
        bound = bound * 2

    return(binary_search_right(L=L, x=x, a=a + bound//2, b=min(a + bound + 1, b)))

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int binary_search_left(int **L,
                            int x,
                            int a,
                            int b):
    cdef int m

    if a >= b:
        return(a)

    if x < L[a][0]:
        return(a)

    if x > L[b-1][0]:
        return(b)

    while a < b:
        m = (a + b) // 2

        if L[m][0] < x:
            a = m + 1
        else:
            b = m

    return(a)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int binary_search_right(int **L,
                             int x,
                             int a,
                             int b):
    cdef int m

    if b <= a:
        return(b)

    if x < L[a][0]:
        return(a)

    if x > L[b-1][0]:
        return(b)

    while a < b:
        m = (a + b) // 2

        if L[m][0] > x:
            b = m
        else:
            a = m +1

    return(b)


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * get_S_shape(int[:,:] U_diff_desc,
                       int n_U,
                       int d):
    cdef Py_ssize_t j

    cdef int *S_shape = <int *> malloc(d * sizeof(int))

    for j in range(d):
        S_shape[j] = 0

        for i_U in range(n_U):
            if U_diff_desc[i_U, j] == 1:
                S_shape[j] = S_shape[j] + 1
    return(S_shape)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double [:] set_estimation(int[:,:] Z_diff_asc,
                                int[:] Z_indices,
                                double[:] g):
    cdef Py_ssize_t i_Z, i_g
    cdef int n_Z = Z_diff_asc.shape[0]
    cdef int d = Z_diff_asc.shape[1]

    cdef double [:] f = np.zeros(n_Z, dtype=np.double)

    i_g = -1
    for i_Z in range(n_Z):
        if Z_diff_asc[i_Z, d-1] == 1:
            i_g = i_g + 1
        f[Z_indices[i_Z]] = g[i_g]
    return(f)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division
def merge(int[:, :] U,
          int[:, :] U_diff_desc,
          double[:] nu,
          int[:, :] Z,
          int q,
          double h,
          int kernel_id,
          double dx,
          int verbose=0):
    cdef Py_ssize_t i_U, i_Z, j, k, i_T

    cdef int n_U = U.shape[0]
    cdef int d = U.shape[1]
    cdef int n_Z = Z.shape[0]

    # get the margin i.e the kernel support radius
    # in cell unit around a kernel center.
    cdef int margin = (q - 1) // 2

    # initialize the output array
    cdef double [:] f = np.zeros(n_Z, dtype=np.double)

    # U sparsing -> S
    cdef int [:,:] U_diff_one_side = count_one_side(A_diff_desc=U_diff_desc)
    cdef int *S_shape = get_S_shape(U_diff_desc,
                                    n_U,
                                    d)
    cdef int ***S = sparse(A=U,
                           A_diff_desc=U_diff_desc,
                           A_diff_one_side = U_diff_one_side,
                           S_shape=S_shape)

    # Z sparsing -> T
    cdef int[:,:] Z_diff_desc = np.ones((n_Z, d), dtype=np.intc)
    count_diff_desc(Z, Z_diff_desc)
    cdef int [:,:] Z_diff_one_side = count_one_side(A_diff_desc=Z_diff_desc)
    cdef int *T_shape = get_S_shape(Z_diff_desc,
                                    n_Z,
                                    d)
    cdef int ***T = sparse(A=Z,
                           A_diff_desc=Z_diff_desc,
                           A_diff_one_side = Z_diff_one_side,
                           S_shape=T_shape)


    # initialize the exploration of the first column (j = 0)
    cdef int low_S = 0
    cdef int high_S = S_shape[0]

    # verbose initialization (tqdm package)
    if verbose > 0:
        for_list = tqdm(range(0, T_shape[0]))
    else:
        for_list = range(0, T_shape[0])

    # exploration of the first column
    # it is a reccursive loop to explore each columns
    for i_T in for_list:
        # the low_S is returned to reduce the S list length
        # at each loop (because T is increasing).

        low_S = explore(S=S,
                        S_shape=S_shape,
                        nu = nu,
                        d=d,
                        T = T,
                        T_shape=T_shape,
                        f=f,
                        margin=margin,
                        j=0,
                        low_S=0,
                        high_S=high_S,
                        i_T=i_T,
                        dx=dx,
                        h=h,
                        kernel_id=kernel_id,
                        dist_sq = 0)

    # free allocated memory
    for j in range(d):
        for i_U in range(S_shape[j]):
            free(S[j][i_U])

        for i_T in range(T_shape[j]):
            free(T[j][i_T])

        free(S[j])
        free(T[j])
    free(S)
    free(T)

    free(S_shape)
    free(T_shape)

    return(np.array(f))

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division
cdef int explore(int ***S,
                 int *S_shape,
                 double[:] nu,
                 int d,
                 int ***T,
                 int *T_shape,
                 double[:] f,
                 int margin,
                 int j,
                 int low_S,
                 int high_S,
                 int i_T,
                 double dx,
                 double h,
                 int kernel_id,
                 double dist_sq):
    cdef Py_ssize_t i_S

    cdef int z = T[j][i_T][0]
    cdef int low_T = T[j][i_T][1]
    cdef int high_T = get_high(T[j], T_shape, d, j, i_T)
    cdef int i_Z
    cdef int a, b

    a = binary_search_left(L = S[j],
                            x = z - margin,
                            a = low_S,
                            b = high_S)
    b = binary_search_right(L = S[j],
                            x = z + margin,
                            a = a,
                            b = high_S)
    # a = exponential_search_left(L = S[j],
    #                             x = z - margin,
    #                             a = low_S,
    #                             b = high_S)
    # b = exponential_search_right(L = S[j],
    #                              x = z + margin,
    #                              a = a,
    #                              b = high_S)

    cdef int low_S_prime = a

    if j < d-1:
        for i_S in range(a, b):
            low_S = S[j][i_S][1]
            high_S = get_high(S[j], S_shape, d, j, i_S)

            if kernel_id > 0:
                dist_sq = dist_sq + cpow(z - S[j][i_S][0], 2.0)

            for i_T in range(low_T, high_T):


                low_S = explore(S=S,
                                S_shape=S_shape,
                                nu = nu,
                                d=d,
                                T=T,
                                T_shape=T_shape,
                                f=f,
                                margin=margin,
                                j=j+1,
                                low_S=low_S,
                                high_S=high_S,
                                i_T = i_T,
                                dx = dx,
                                h = h,
                                kernel_id = kernel_id,
                                dist_sq = dist_sq)
    else:
        i_Z = T[j][i_T][1]

        if kernel_id == 0:
            # box kernel
            for i_S in range(a, b):
                f[i_Z] = f[i_Z] + nu[S[j][i_S][1]]

        elif kernel_id == 1:
            # gaussian kernel
            for i_S in range(a, b):
                dist_sq = ( dist_sq + cpow(z - S[j][i_S][0], 2.0) ) * cpow(dx, 2.0)

                f[i_Z] = f[i_Z] + nu[S[j][i_S][1]] * exp(-dist_sq / cpow(h,2.0) / 2)

        elif kernel_id == 2:
            for i_S in range(a, b):
                dist_sq = ( dist_sq + cpow(z - S[j][i_S][0], 2.0) ) * cpow(dx, 2.0)

                if dist_sq < 1:
                    f[i_Z] = f[i_Z] + nu[S[j][i_S][1]] * 0.75 * (1 - dist_sq) ** 2


    return(low_S_prime)


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_high(int** S_j,
              int* S_shape,
              int d,
              int j,
              int i):

    if i == S_shape[j] - 1:
        if j == d - 1:
            return(S_j[i][1] + 1)
        else:
            return(S_shape[j+1])
    else:
        return(S_j[i+1][1])

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division
cpdef double[:] boundary_correction_gaussian_kernel(double[:,:] A,
                                          double[:] R,
                                          double[:,:] C,
                                          double h,
                                          int n_mc=10000,
                                          double zero=0.000000000001):
    cdef Py_ssize_t i, i_hyp, j, i_mc

    cdef int d = A.shape[0]
    cdef int n_hyp = A.shape[1]
    cdef int n = C.shape[0]

    cdef double normalization = (2 * np.pi)**(d/2) * h**d

    cdef double [:] norm_2_a_square = np.linalg.norm(A, axis=0, ord=2)**2
    cdef double [:] norm_inf_a = np.linalg.norm(A, axis=0, ord=np.inf)

    cdef double[:,:] X_mc = np.random.random((n_mc, d))

    cdef double[:] g = np.zeros(n_mc, dtype=np.double)

    cdef double dist_sq
    for i_mc in range(n_mc):
        dist_sq = 0.0
        for j in range(d):
            dist_sq = dist_sq + cpow(X_mc[i_mc,j] - 0.5, 2.0)

        g[i_mc] = exp(-dist_sq / cpow(h,2.0) / 2) / normalization


    cdef double[:] V = np.ones(n, dtype=np.double)

    cdef double dist

    cdef double dist_max = h * 3.0

    cdef double [:,:] A_x = np.zeros((d, n_hyp), dtype=np.double)
    cdef double [:] R_x = np.zeros(n_hyp, dtype = np.double)
    cdef int m_x

    cdef double s

    for i in range(n):
        m_x = 0
        # distances to hyperplanes
        for i_hyp in range(n_hyp):
            dist = R[i_hyp]
            for j in range(d):
                dist = dist + C[i, j] * A[j, i_hyp]
            dist = abs(dist) / norm_2_a_square[i_hyp] * norm_inf_a[i_hyp]

            if dist <= dist_max:
                for j in range(d):
                    A_x[j, m_x] = A[j, i_hyp]
                R_x[m_x] = R[i_hyp]

                # affine transformation
                for j in range(d):
                    R_x[m_x] = R_x[m_x] - A_x[j, m_x] * (-C[i,j] + 3 * h)
                    A_x[j, m_x] = A_x[j, m_x] * 6 * h

                m_x = m_x + 1


        if m_x > 0:
            # count side
            s = 0
            for i_mc in range(n_mc):

                for i_hyp in range(m_x):
                    dist = zero + R_x[i_hyp]
                    for j in range(d):
                        dist = dist + X_mc[i_mc, j] * A_x[j, i_hyp]

                    if dist <= 0:
                        break
                else:
                    s = s + g[i_mc]

            V[i] = s / n_mc

            if V[i] <= zero:
                V[i] = 1.0

    return(V)
