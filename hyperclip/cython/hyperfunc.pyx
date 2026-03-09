# distutils: language = c++

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const void *, const void *)) nogil

# from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool

import cython

import numpy as np
cimport numpy as np

cpdef double volume(double[:, :] A,
                   double[:] R,
                   bool check_A = True,
                   double zero = 0.0000001):
    
    cdef int m = A.shape[1]
    
    return(volume_according_m(A = A, 
                              R = R,
                              m = m,
                              check_A = check_A,
                              zero = zero))

cpdef double [:] volumes(double[:, :] A,
                         double[:] R,
                         double [:, :] X,
                         double h,
                         double zero = 0.0000001):
    
    cdef Py_ssize_t i, i_hyp, j
    
    cdef int N = X.shape[0]
    
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    
    cdef double [:] vols = np.ones(N, dtype=np.double)
    
    cdef double [:,:] A_x = np.zeros((n, m), dtype=np.double)
    cdef double [:] R_x = np.zeros(n, dtype = np.double)
        
    cdef double [:] x = np.zeros(n, dtype=np.double)
    cdef double *dist_to_hyp
    
    cdef double [:] norm_inf_w = np.linalg.norm(A, axis=0, ord=np.inf)
    cdef double [:] norm_2_w = np.linalg.norm(A, axis=0, ord=2)
    
    cdef int m_x
    
    for i in range(N):
        for j in range(n):
            x[j] = X[i, j]
        dist_to_hyp = inf_distances_to_hyperplanes(A = A,
                                                   R = R,
                                                   x = x,
                                                   norm_inf_w = norm_inf_w,
                                                   norm_2_w = norm_2_w)
        
        m_x = 0
        for i_hyp in range(m):
            if dist_to_hyp[i_hyp] <= h / 2.0:
                for j in range(n):
                    A_x[j, m_x] = A[j, i_hyp]
                R_x[m_x] = R[i_hyp]
                
                # affine transformation
                for j in range(n):
                    R_x[m_x] = R_x[m_x] - A_x[j, m_x] * (-x[j] + h / 2)
                    A_x[j, m_x] = A_x[j, m_x] * h
                
                m_x = m_x + 1
                  
        if m_x > 0:
            vols[i] = volume_according_m(A=A_x,
                                         R=R_x,
                                         m=m_x,
                                         check_A=False,
                                         zero=zero)
        
        free(dist_to_hyp)
        
    return(vols)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Deactivate zero division checking.
cdef double * inf_distances_to_hyperplanes(double[:, :] A, 
                                           double[:] R,
                                           double[:] x,
                                           double[:] norm_inf_w,
                                           double[:] norm_2_w):
    
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    
    cdef double * dist
    dist = <double *> malloc(m * sizeof(double))
    cdef Py_ssize_t i_hyp, j
    
    # initialize dist_min greater than the limit condition
    for i_hyp in range(m):
        # distance to hyperplan
        dist[i_hyp] = 0.0
        for j in range(n):
            dist[i_hyp] = dist[i_hyp] + x[j] * A[j, i_hyp]
        dist[i_hyp] = dist[i_hyp] + R[i_hyp]
        # dist[i_hyp] = abs(dist[i_hyp]) / norm_2_w[i_hyp]
        dist[i_hyp] = abs(dist[i_hyp]) / norm_2_w[i_hyp]**2 * norm_inf_w[i_hyp]
        
    return(dist)

cpdef double volume_according_m(double[:, :] A,
                                double[:] R,
                                int m,
                                bool check_A = True,
                                double zero = 0.0000001):
    if check_A:
        if not clipping_condition_A_according_m(A=A, R=R, m=m, zero=zero):
            # print('Error : clipping condition A unsatisfied. Return 1.0 as volume')
            return(1.0)
    
    return(clipping_condition_B_and_volume(A = A, 
                                           R = R,
                                           m = m,
                                           return_volume = True,
                                           zero = zero))

cpdef bool clipping_condition_A(double[:, :] A, 
                                double[:] R,
                                double zero = 0.0000001):
    cdef int m = A.shape[1]
    return(clipping_condition_A_according_m(A=A, 
                                            R=R,
                                            m=m,
                                            zero=zero))

cpdef bool clipping_condition_B(double[:, :] A, 
                                double[:] R,
                                double zero = 0.0000001):
    cdef int m = A.shape[1]
    
    return(clipping_condition_B_according_m(A=A, 
                                            R=R,
                                            m=m,
                                            zero = zero))

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef bool clipping_condition_A_according_m(double[:, :] A, 
                                            double[:] R,
                                            int m,
                                            double zero = 0.0000001):
    cdef Py_ssize_t card_I, q_I, q_K, r_J, q_J, i
    
    cdef int n = A.shape[0]
    
    cdef int card_list_I
    cdef int **list_I, **list_J_indices, **list_K
    cdef int *I, *J, *K, *K_bar
    cdef int card_J, card_K
    
    cdef int pt_size = max(n+1, m-1)
    cdef int **pt = pascal_triangle(pt_size)
    
    cdef double *v
    
    for card_I in range((m-1) + 1):
        
        list_I = combinations_indices(m-1, card_I, pt)
        card_list_I = binomial_coefficient_in_pascal_triangle(m-1, card_I, pt)
        
        for q_I in range(card_list_I):
            I = list_I[q_I]
            
            card_K = card_I - 1
            list_K = combinations_indices(n, card_K, pt)
                            
            for q_K in range(binomial_coefficient_in_pascal_triangle(n, card_K, pt)):
                K = list_K[q_K]
                K_bar = bar(n, K, card_K, sorted_lst=True)
                                
                for card_J in range( n - card_K + 1):
                    
                    list_J_indices = combinations_indices(n - card_K, card_J, pt)
                    
                    J = <int *> malloc(card_J * sizeof(int))
                    # /!\ flag : possible de devoir mettre J à 0
                    
                    for q_J in range(binomial_coefficient_in_pascal_triangle(n - card_K, card_J, pt)):
                        for r_J in range(card_J):
                            J[r_J] = K_bar[list_J_indices[q_J][r_J]]
                            
                        v = compute_vertex(A=A, 
                                           R=R, 
                                           I=I, 
                                           card_I=card_I,
                                           J=J, 
                                           card_J=card_J,
                                           K=K, 
                                           card_K=card_K,
                                           zero=zero)
                                                
                        if test_vertex(A, R, m, I, card_I, v, zero * 10):
                            # one vertex is solution !
                            # according to the clipping condition (A) definition
                            # the condition is then unsatisfied
                            # False is therefore returned
                            
                            return(False)
                        
                        free(list_J_indices[q_J])
                        free(v)
                    free(J)
                    free(list_J_indices)
                free(K)
                free(K_bar)
            free(list_K)
            free(I)
        free(list_I)
    free_int_2d(pt, pt_size)
    
    return(True)

cpdef bool clipping_condition_B_according_m(double[:, :] A, 
                                            double[:] R,
                                            int m,
                                            double zero = 0.0000001):
        
    if clipping_condition_B_and_volume(A = A, 
                                       R = R,
                                       m=m,
                                       return_volume = False,
                                       zero = zero) == 1.0:
        return(True)
    else:
        return(False)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Deactivate zero division checking.
cdef double clipping_condition_B_and_volume(double[:, :] A, 
                                            double[:] R,
                                            int m,
                                            bool return_volume=True,
                                            double zero = 0.0000001):
    cdef Py_ssize_t card_I, q_I, q_J, q_K, r_J, id_t, i, j
    
    cdef int n = A.shape[0]
    
    cdef int pt_size = max(n+1, m-1)
    cdef int **pt = pascal_triangle(pt_size)
    
    cdef int **list_I, **list_J_indices, **list_K
    cdef int card_list_I, card_list_J_indices, card_list_K
    cdef int *I, *J, *K, *K_bar
    cdef int card_J, card_K
    
    cdef int *I_union_m, *I_union_m_remove_t
    
    cdef double *v
    cdef int *v_01, *v_star, *v_star_union_t
    cdef int n_01, n_0
    cdef int v_star_sum
    
    cdef double numerator, denominator, A_v_star_I, g_m_v
    
    cdef double vol = 0.0
    
    for card_I in range((m -1) + 1):
        list_I = combinations_indices(m-1, card_I, pt)
        card_list_I = binomial_coefficient_in_pascal_triangle(m-1, card_I, pt)
        
        for q_I in range(card_list_I):
            I = list_I[q_I]
            
            card_K = card_I
            list_K = combinations_indices(n, card_K, pt)
            card_list_K = binomial_coefficient_in_pascal_triangle(n, card_K, pt)
            for q_K in range(card_list_K):
                K = list_K[q_K]
                K_bar = bar(n, K, card_K, sorted_lst=True)
                
                for card_J in range( n - card_K + 1):
                    
                    list_J_indices = combinations_indices(n - card_K, card_J, pt)
                    card_list_J_indices = binomial_coefficient_in_pascal_triangle(n - card_K, card_J, pt)
                    J = <int *> malloc(card_J * sizeof(int))
                    # /!\ flag : possible de devoir mettre J à 0
                    for q_J in range(card_list_J_indices):
                        
                        for r_J in range(card_J):
                            J[r_J] = K_bar[list_J_indices[q_J][r_J]]
                        v = compute_vertex(A=A, 
                                            R=R, 
                                            I=I, 
                                            card_I=card_I,
                                            J=J, 
                                            card_J=card_J,
                                            K=K, 
                                            card_K=card_K,
                                            zero=zero)
                        
                        if test_vertex(A, R, m, I, card_I, v, zero * 10):
                            n_01 = count_01(v, n)
                            v_01 = get_vertex_01(v, n, n_01)
                            
                            v_star = bar(n, v_01, n_01, sorted_lst=True)
                            
                            denominator = 1.0
                            for id_t in range(card_I):
                                I_union_m_remove_t = union_remove(I = I, 
                                                                  card_I = card_I,
                                                                  m = m-1, 
                                                                  t = I[id_t])
                                                                
                                denominator = denominator * get_det_sub_A(
                                    A = A, 
                                    I = v_star, 
                                    card_I = n - n_01,
                                    J = I_union_m_remove_t,
                                    card_J = card_I)
                                
                                free(I_union_m_remove_t)
                            
                            for id_t in range(n_01):
                                I_union_m = union(I=I,
                                                  card_I=card_I,
                                                  a = m-1)
                                
                                v_star_union_t = union(I = v_star,
                                                        card_I=n - n_01,
                                                        a = v_01[id_t])
                                                                
                                denominator = denominator * get_det_sub_A(
                                    A = A, 
                                    I = v_star_union_t, 
                                    card_I = n - n_01 + 1,
                                    J = I_union_m,
                                    card_J = card_I + 1)
                                
                                free(I_union_m)
                                free(v_star_union_t)
                            
                            free(v_01)
                            
                                
                            if abs(denominator) < zero:
                                if not return_volume:
                                    return(0.0)
                                else:
                                    # print('Error : clipping condition B unsatisfied. Return 1.0 as volume')
                                    return(1.0)
                            
                            if return_volume:
                                n_0 = count_0(v, n)
                                
                                v_star_sum = 0
                                for i in range(n - n_01):
                                    v_star_sum = v_star_sum + v_star[i] + 1
                                
                                g_m_v = R[m-1]
                                for i in range(n):
                                    g_m_v = g_m_v + A[i,m-1] * v[i]
                                
                                A_v_star_I = get_det_sub_A(A = A, 
                                                           I = v_star, 
                                                           card_I = n - n_01,
                                                           J = I,
                                                           card_J = card_I)
                                
                                free(v_star)
                                
                                numerator = (-1)**(n_0 + v_star_sum)
                                numerator = numerator * (g_m_v * A_v_star_I)**n
                                
                                denominator = denominator * factorial(n) * abs(A_v_star_I)
                                vol = vol + numerator / denominator
                    
                        free(list_J_indices[q_J])
                        free(v)
                    free(J)    
                    free(list_J_indices)
                free(K)
                free(K_bar)
            free(list_K)
            free(I)
        free(list_I)
    free_int_2d(pt, pt_size)
    
    if return_volume:
        return(vol)
    else:
        return(1.0) # clipping condition B satisfied.

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void getCofactor(double **A, 
                      double **temp, 
                      int p,
                      int q, 
                      int n):
    cdef int i = 0
    cdef int j = 0
 
    # Looping for each element of the matrix
    for row in range(n):
        for col in range(n):
            #  Copying into temporary matrix only those
            #  element which are not in given row and
            #  column
            if row != p and col != q:
                
                temp[i][j] = A[row][col];
                j = j + 1
 
                # Row is filled, so increase row index and
                # reset col index
                if j == n - 1:
                    j = 0
                    i = i + 1
 
# Recursive function for finding determinant of matrix.
#   n is current dimension of A[,].
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double determinant(double **A, 
                        int n):
    # inspired by https://www.geeksforgeeks.org/determinant-of-a-matrix/
    
    cdef double D = 0 # Initialize result
    
    #  Base cases
    if n == 0:
        return(1.0)
    elif n == 1:
        return A[0][0]
    
    # To store cofactors
    cdef double **temp = <double **> malloc(n * sizeof(double **))
    for i in range(n):
        temp[i] = <double *> malloc(n * sizeof(double *))
    
    cdef int sign = 1 # To store sign multiplier
    
    # Iterate for each element of first row
    for f in range(n):
        # Getting Cofactor of A[0,f]
        getCofactor(A, temp, 0, f, n)
        
        D = D + sign * A[0][f] * determinant(temp, n - 1)
     
        # terms are to be added with alternate sign
        sign = -sign
    
    free_double_2d(temp, n)
    
    return D

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double get_det_sub_A(double[:, :] A,
                          int *I,
                          int card_I,
                          int *J,
                          int card_J):
    
    cdef double **sub_A
    cdef double det
    sub_A = get_sub_A(A = A,
                      I = I,
                      card_I = card_I,
                      J = J,
                      card_J = card_J)
    
    det = determinant(A = sub_A, 
                       n = card_I)
    
    free_double_2d(sub_A, card_I)
    
    return(det)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double ** get_sub_A(double[:, :] A,
                         int *I,
                         int card_I,
                         int *J,
                         int card_J):
    cdef Py_ssize_t id_I, id_J
    
    cdef double **B = <double **> malloc(card_I * sizeof(double *))
    
    for id_I in range(card_I):
        B[id_I] = <double *> malloc(card_J * sizeof(double))
        for id_J in range(card_J):
            B[id_I][id_J] = A[I[id_I],J[id_J]]
    
    return(B)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * union(int *I,
                   int card_I,
                   int a):
    # union in the increasing order
    cdef Py_ssize_t i
    
    cdef int *I_u = <int *> malloc((card_I + 1)*sizeof(int))
    cdef int shift
    
    shift = 0
    for i in range(card_I):
        if I[i] > a and shift == 0:
            I_u[i] = a
            shift = 1
        I_u[i + shift] = I[i]
    
    if shift == 0:
        I_u[card_I] = a
    
    return(I_u)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * union_remove(int *I,
                          int card_I,
                          int m,
                          int t):
    # nor order expectation
    # because it is always used in a right way.
    cdef Py_ssize_t i
    
    cdef int *I_ur = <int *> malloc(card_I * sizeof(int))
    cdef int shift
        
    shift = 0
    for i in range(card_I-1):
        if I[i] == t:
            shift = 1
        I_ur[i] = I[i + shift]
    
    I_ur[card_I-1] = m
    
    return(I_ur)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int count_01(double *v, int n):
    cdef Py_ssize_t i
    cdef int n_01 = 0
    
    for i in range(n):
        if v[i] == 0 or v[i] == 1:
            n_01 = n_01 + 1
    return(n_01)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * get_vertex_01(double *v,
                         int n,
                         int n_01):
    cdef Py_ssize_t i
    cdef int i_v_01
    cdef int *v_01 = <int *> malloc(n_01 * sizeof(int))
        
    i_v_01 = 0
    for i in range(n):
        if v[i] == 0 or v[i] == 1:
            v_01[i_v_01] = i
            i_v_01 = i_v_01 + 1
            
    return(v_01)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int count_0(double *v, int n):
    cdef Py_ssize_t i
    cdef int n_0 = 0
    
    for i in range(n):
        if v[i] == 0:
            n_0 = n_0 + 1
    return(n_0)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * get_vertex_0(double *v,
                        int n,
                        int n_0):
    cdef Py_ssize_t i
    cdef int i_v_0
    cdef int *v_0 = <int *> malloc(n_0 * sizeof(int))
        
    i_v_0 = 0
    for i in range(n):
        if v[i] == 0:
            v_0[i_v_0] = i
            i_v_0 = i_v_0 + 1
            
    return(v_0)

# @cython.boundscheck(False)  # Deactivate bounds checking.
# @cython.wraparound(False)   # Deactivate negative indexing.
# cdef double ** convert_double_matrix_numpy_c(double[:, :] A):
#     cdef Py_ssize_t i, j
#     cdef int n = A.shape[0]
#     cdef int m = A.shape[1]
#     cdef double **A_c
    
#     A_c = <double **> malloc(n * sizeof(double **))
        
#     for i in range(n):
#         A_c[i] = <double *> malloc(m * sizeof(double *))
#         for j in range(m):
#             A_c[i,j] = A[i,j]
    
#     return(A_c)

# @cython.boundscheck(False)  # Deactivate bounds checking.
# @cython.wraparound(False)   # Deactivate negative indexing.
# cdef double * convert_double_vector_numpy_c(double[:] V):
#     cdef Py_ssize_t i
#     cdef int n = V.shape[0]
#     cdef double *V_c
    
#     V_c = <double *> malloc(n * sizeof(double *))
        
#     for i in range(n):
#         V_c[i] = V[i]
    
#     return(V_c)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double * compute_vertex(double[:, :] A, 
                               double[:] R, 
                               int *I,
                               int card_I,
                               int *J,
                               int card_J,
                               int *K,
                               int card_K,
                               double zero = 0.0000001):
    
    cdef Py_ssize_t i, id_I, id_J, id_K
    
    cdef int n = A.shape[0]
    
    cdef double *v = <double *> malloc(n * sizeof(double))
    # generate system
    # A_sys is directly transposed
    # in order to correspond to the classical gaussian solving A_sys X = B_sys.
    # we should have card_I >= card_K
    # to prevent card_I > card_K which lead to unsolvable system
    # we limit to card_I equations to have (card_K, card_K) system.
    
    cdef double **AB_sys = <double **> malloc(card_I * sizeof(double **))
    
    for id_I in range(card_I):
        AB_sys[id_I] = <double *> malloc((card_K + 1) * sizeof(double *))
        for id_K in range(card_K):
            AB_sys[id_I][id_K] = A[K[id_K],I[id_I]]
        
        AB_sys[id_I][card_K] = - R[I[id_I]]
        for id_J in range(card_J):
            AB_sys[id_I][card_K] = AB_sys[id_I][card_K] - A[J[id_J],I[id_I]]
    # we have now A_sys a (card_K, card_K) matrix and B_sys a (card_K,) vector
    # fusioned in an AB_sys matrix
    
    # solve
    cdef double *v_K = gauss_AB(AB=AB_sys, 
                                n=card_K, # nb of variables. 
                                zero=zero)
    
    for i in range(n):
        v[i] = 0
    
    for id_K in range(card_K):
        v[K[id_K]] = v_K[id_K]
        
    for id_J in range(card_J):
        v[J[id_J]] = 1    
    
    free_double_2d(AB_sys, card_I)
    free(v_K)
    return(v)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef bool test_vertex(double[:, :] A,
                      double[:] R,
                      int m,
                      int *I,
                      int card_I,
                      double *v,
                      double zero = 0.000001):
    cdef Py_ssize_t id_I, i
    
    cdef int n = A.shape[0]
    
    cdef double g
    
    cdef int *I_bar = bar(m, I, card_I, sorted_lst=True)
    
    # H = 
    for id_I in range(card_I):
        g = R[I[id_I]]
        for i in range(n):
            g = g + A[i,I[id_I]] * v[i]
        # print('g=', g)
        if abs(g) > zero: # 0.000001
            free(I_bar)
            return(False)
        
    # H +
    for id_I in range(m - card_I):
        g = R[I_bar[id_I]]
        for i in range(n):
            g = g + A[i,I_bar[id_I]] * v[i]
        
        if g < 0:
            free(I_bar)
            return(False)
    
    # Hypercube
    for i in range(n):
        if v[i] < 0 or v[i] > 1:
            free(I_bar)
            return(False)
    
    free(I_bar)
    return(True)
    
cdef double * gauss(double **A,
                    double *B,
                    int n,
                    double zero=0.0000001):
   
    
    cdef double **AB = generate_AB(A, B, n)
    
    cdef double *x = gauss_AB(AB=AB, n=n, zero=zero)
    
    free_double_2d(AB, n)
    
    return x

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Deactivate zero division checking.
cdef double * gauss_AB(double **AB,
                       int n,
                       double zero=0.0000001):
    # inspired by https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    cdef Py_ssize_t i, j, k
    
    # note that variables are along the axis 1 of A !
    
    cdef double c
    cdef double *x = <double *> malloc(n * sizeof(double))
    cdef double maxEl, tmp
    cdef int maxRow
        
    for i in range(n):
        # Search for maximum in this column
        maxEl = abs(AB[i][i])
        maxRow = i
        for k in range(i+1,n):
            if abs(AB[k][i]) > maxEl:
                maxEl = abs(AB[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = AB[maxRow][k]
            AB[maxRow][k] = AB[i][k]
            AB[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            if AB[i][i] == 0.0:
                AB[i][i] = zero
                # print('1', i, AB[k,i], AB[i,i])
                
            c = -AB[k][i]/AB[i][i]
            for j in range(i, n+1):
                if i==j:
                    AB[k][j] = 0;
                else:
                    AB[k][j] += c * AB[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    for i in range(n - 1, -1, -1):
        if AB[i][i] == 0.0:
            # print('2', i, AB[i,n], AB[i,i])
            AB[i][i] = zero
            
            
        x[i] = AB[i][n]/AB[i][i]
        for k in range(i - 1, -1, -1):
            AB[k][n] -= AB[k][i] * x[i]
        
    return x

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double ** generate_AB(double **A,
                           double *B,
                           int n):
    cdef Py_ssize_t i, j
    cdef double **AB = <double **> malloc(n*sizeof(double *))
    
    for i in range(n):
        AB[i] = <double *> malloc((n+1) *sizeof(double))
        for j in range(n):
            AB[i][j] = A[i][j]
        
        AB[i][n] = B[i]
    
    return(AB)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * bar(int n, 
               int *K,
               int card_K,
               bool sorted_lst=False):
    cdef Py_ssize_t i, j_K
    
    cdef int j_K_bar
    cdef bool trigger
    
    cdef int *K_bar = <int *> malloc((n - card_K) * sizeof(int))
    
    j_K_bar = 0
    for i in range(n):
        trigger = True
        for j_K in range(card_K):
            if K[j_K] == i:
                trigger = False
                if sorted_lst:
                    break
        if trigger:
            K_bar[j_K_bar] = i
            j_K_bar = j_K_bar + 1
    
    return(K_bar)
           
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int ** combinations_indices(int n, 
                                 int k, 
                                 int **pt):
    cdef Py_ssize_t i, j
    cdef int *indices
    cdef int id_ci, bc
    
    # compute the binomial coefficient
    # i.e. the number of combinations
    bc = binomial_coefficient_in_pascal_triangle(n, k, pt)
    
    if k < 0 or n - k < 0:
        return(<int **> malloc(0 * sizeof(int *)))
    
    # cdef int [:, :] ci = np.zeros((bc, k), dtype=np.intc)
    
    cdef int ** ci = <int **> malloc(bc * sizeof(int *))
    
    if bc == 0:
        return(ci)
    
    
    indices = list_range(k)
    
    id_ci = 0
    ci[id_ci] = <int *> malloc(k * sizeof(int))
    for i in range(k):
        ci[id_ci][i] = indices[i] + 0
    id_ci = id_ci + 1

    while True:
        for i in reversed(range(k)):
            if indices[i] != i + n - k:
                break
        else:
            free(indices)
            return ci
        indices[i] += 1
        for j in range(i+1, k):
            indices[j] = indices[j-1] + 1
        
        ci[id_ci] = <int *> malloc(k * sizeof(int))
        for i in range(k):
            ci[id_ci][i] = indices[i] + 0
        id_ci = id_ci + 1

cdef int factorial(int n):
    cdef int f = 1
    cdef Py_ssize_t i
    
    for i in range(1,n+1):
        f = f * i
    
    return(f)

@cython.cdivision(True)
cdef int binomial_coefficient(int n, int k):
    if k < 0 or n-k < 0:
        return(0)
    
    return(factorial(n) / (factorial(k) * factorial(n-k)))

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int ** pascal_triangle(int n):
    cdef Py_ssize_t i, j
    # cdef int[:, :] pt = np.zeros((n+1, n+1), dtype=np.intc)
    
    cdef int **pt = <int **> malloc((n+1) * sizeof(int *))
    
    # make diag and first column to 1
    for i in range(n+1):
        pt[i] = <int *> malloc((n+1) * sizeof(int))
        pt[i][i] = 1
        pt[i][0] = 1
        
    for i in range(1, n+1):
        for j in range(1, i):
            pt[i][j] = pt[i-1][j-1] + pt[i-1][j] 
    
    return(pt)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int binomial_coefficient_in_pascal_triangle(int n, 
                                                 int k,
                                                 int **pt):
    if k < 0 or n-k < 0:
        return(0)
    return(pt[n][k])

cdef int * list_range(int n):
    cdef Py_ssize_t i
    
    cdef int *lst = <int *> malloc(n * sizeof(int))
    
    for i in range(n):
        lst[i] = i
    
    return(lst)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void free_int_2d(int **A, 
                      int n):
    cdef Py_ssize_t i
    for i in range(n):
        free(A[i])
    free(A)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void free_double_2d(double **A, 
                      int n):
    cdef Py_ssize_t i
    for i in range(n):
        free(A[i])
    free(A)