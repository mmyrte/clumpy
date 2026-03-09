import numpy as np

import ekde.ekdefunc
from ekde.ekdefunc import merge as ekdefunc_merge
import hyperclip.hyperfunc
from hyperclip import Hyperplane
from .whitening_transformer import WhiteningTransformer
import pandas as pd
from scipy.special import gamma, factorial2
from scipy.stats import norm
from joblib import dump, load
import time
import os
import shutil
from joblib import wrap_non_picklable_objects
from multiprocessing import Pool

kernels_id = {'box' : 0,
              'gaussian' : 1,
              'epa' : 2}

supports = {'box' : 1.0,
            'gaussian' : 3.0,
            'epa' : 1.0}

R = {'box' : 1/2,
     'gaussian' : 1 / (2 * np.sqrt(np.pi)),
     'epa' : 3 / 5}

def compute_centers(Z, x_min, dx):
    C = x_min + Z * dx + 0.5 * dx
    return(C)

class KDE():
    def __init__(self, 
                 h='terrel', 
                 kernel='box',
                 q=51, 
                 bounds=[],
                 n_jobs = 1,
                 zero = 0.00000000000001,
                 n_mc_axes = 100,
                 wt = True,
                 verbose=0):
        if q%2 == 0:
            raise(ValueError("Unexpected q value. q should be an odd int."))
                
        self.h = h
        self.kernel = kernel
        self._h = None
        self.q = q
        self.bounds = bounds
        self.n_jobs = n_jobs
        self.zero = zero
        self.n_mc_axes = n_mc_axes
        self.wt = wt
        self.verbose = verbose
    
    def fit(self, X):
        if self.verbose > 0:
            print('KDE fit process, X.shape=', X.shape)
        # preprocessing
        if len(X.shape) == 1:
            X = X[:,None]
        
        self._real_X_min = np.min(X, axis=0)
        self._real_X_max = np.max(X, axis=0)
        
        # get data dimensions
        self._n, self._d = X.shape
        
        # find a point not on boundaries
        id_x = self._find_point_not_on_boundaries(X)
        
        # preprocessing
        if self.wt:
            self._wt = WhiteningTransformer()
            X = self._wt.fit_transform(X)
        
        self._x_min = np.min(X, axis=0)
        
        # BOUNDARIES INFORMATIONS
        A, R = self._set_boundaries(x = X[id_x])
        self.A = A
        self.R = R
        # BANDWIDTH SELECTION
        self._compute_bandwidth(X)
        
        self._support = supports[self.kernel]
        
        self._x_min = X.min(axis=0)
        self._dx = self._h * 2 * self._support / self.q
        
        Z = self._discretize(X).astype(np.intc)
        Z = pd.DataFrame(Z)
        
        U_nu = Z.groupby(by=Z.columns.tolist()).size().reset_index(name="nu")
        
        self._U = U_nu[Z.columns.tolist()].values.astype(np.intc)
        self._nu = U_nu["nu"].values.astype(np.double)
        
        self._U_diff_desc = np.ones((self._U.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_desc(self._U, self._U_diff_desc)
        
        C = compute_centers(self._U, self._x_min, self._dx).astype(np.double)
        
        if self.kernel == 'box':
            self._nu /= np.array(hyperclip.hyperfunc.volumes(A, 
                                                             R, 
                                                             C, 
                                                             self._h, 
                                                             zero=self.zero))
        elif self.kernel == 'gaussian':
            self._nu /= np.array(ekde.ekdefunc.boundary_correction_gaussian_kernel(A = A, 
                R = R, 
                C = C, 
                h = self._h, 
                n_mc=self.n_mc_axes**self._d,
                zero=self.zero))
        
        self._compute_normalization()
        
        return(self)
    
    def predict(self, X):
        if self.wt:
            X = self._wt.transform(X)
        
        id_out_of_bounds = np.zeros(X.shape[0]).astype(np.bool)
        for hyp in self._bounds_hyperplanes:
            id_out_of_bounds = np.any((id_out_of_bounds, ~hyp.side(X)), axis=0)
        Z = self._discretize(X)
        
        # SORT Z
        # ======
        Z = pd.DataFrame(Z)
        # keep the index safe
        Z['j'] = np.arange(Z.shape[0])
        # sort by i 
        Z = Z.sort_values(by=[i for i in range(self._d)])
        Z_indices = Z['j'].values.astype(np.intc)
        Z = Z[[i for i in range(self._d)]].values.astype(np.intc)
        
        Z_diff_asc = np.ones((Z.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_asc(Z, Z_diff_asc)
        
        id_Z_unique = np.where(Z_diff_asc[:, self._d-1] == 1)[0]
        
        streams = [int(id_Z_unique.size / self.n_jobs) * i for i in range(self.n_jobs + 1)]
        streams.append(id_Z_unique.size)
        
        folder = './.temp_joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        
        Z_filename_memmap = os.path.join(folder, 'Z_memmap')
        dump(Z, Z_filename_memmap)
        Z_memmap = load(Z_filename_memmap, mmap_mode='r')
        
        U_filename_memmap = os.path.join(folder, 'U_memmap')
        dump(self._U, U_filename_memmap)
        U_memmap = load(U_filename_memmap, mmap_mode='r')
        
        U_diff_desc_filename_memmap = os.path.join(folder, 'U_diff_desc_memmap')
        dump(self._U_diff_desc, U_diff_desc_filename_memmap)
        U_diff_desc_memmap = load(U_diff_desc_filename_memmap, mmap_mode='r')
        
        nu_filename_memmap = os.path.join(folder, 'nu_memmap')
        dump(self._nu, nu_filename_memmap)
        nu_memmap = load(nu_filename_memmap, mmap_mode='r')
        
        pool = Pool(processes=self.n_jobs)
        g = pool.starmap(ekde.ekdefunc.merge, 
                          [(U_memmap,
                            U_diff_desc_memmap,
                            nu_memmap,
                            Z_memmap[id_Z_unique[streams[i_stream]: streams[i_stream+1]]],
                            self.q,
                            self._h,
                            kernels_id[self.kernel],
                            self._dx,
                            self.verbose) for i_stream in range(self.n_jobs + 1)])
        
        pool.close()
        
        try:
            shutil.rmtree(folder)
        except:  # noqa
            print('Could not clean-up automatically.')
        
        g = np.hstack([gi for gi in g])
        
        
        f = np.array(ekde.ekdefunc.set_estimation(Z_diff_asc=Z_diff_asc,
                                                  Z_indices=Z_indices,
                                                  g=g))
        
        f[id_out_of_bounds] = 0.0
        
        f = f / self._normalization
        
        if self.wt:
            f /= self._wt.scale_
            
        return(f)
    
    def _discretize(self, X):
        Z = ((X - self._x_min) / self._dx).astype(int)
        return(Z)
    
    def _find_point_not_on_boundaries(self, X):
        trigger_point_found = True
        
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        
        while trigger_point_found:
            id_x = np.random.choice(self._n)
            x = X[id_x]
            
            if np.all(X_min < x) and np.all(x < X_max):
                trigger_point_found = False
                return(id_x)
        
        
    
    def _set_boundaries(self, x):
        self._bounds_hyperplanes = []
        
        if type(self.bounds) is list:
            for k, pos in enumerate(self.bounds):
                if pos == 'left':
                    self._add_boundary(k=k,
                                       value=self._real_X_min[k],
                                       x=x)
                elif pos == 'right':
                    self._add_boundary(k=k,
                                       value=self._real_X_max[k],
                                       x=x)
                elif pos == 'both':
                    self._add_boundary(k=k,
                                       value=self._real_X_min[k],
                                       x=x)
                    self._add_boundary(k=k,
                                       value=self._real_X_max[k],
                                       x=x)
                # else:
                    # raise(TypeError('Unexpected bounds parameters'))
        
        A = np.zeros((self._d, len(self._bounds_hyperplanes)))
        R = np.zeros(len(self._bounds_hyperplanes))
        
        for i_hyp, hyp in enumerate(self._bounds_hyperplanes):
            A[:, i_hyp] = hyp.a
            R[i_hyp] = hyp.r
        
        return(A, R)
    
    def _add_boundary(self, k, value, x):
        P = np.diag(np.ones(self._d))
        
        P[:, k] = value
        
        if self.wt:
            P = self._wt.transform(P)

        hyp = Hyperplane().set_by_points(P)
        hyp.set_positive_side(x)
        self._bounds_hyperplanes.append(hyp)
    
    def _compute_bandwidth(self, X):
        if type(self.h) is int or type(self.h) is float or type(self.h) is np.float64:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                # the scott rule is based on gaussian kernel
                
                self._h = scott_bandwidth(n=X.shape[0],
                                          d=X.shape[1])
                
                if self.kernel == 'box':
                    # the support of the gaussian kernel to have 99%
                    # of the density is (2.576/2) = 1.288
                    self._h *= 1.288
                elif self.kernel == 'gaussian':
                    self._h *= 4
            
            elif self.h == 'terrel':
                self._h = terrel_bandwidth(n=X.shape[0], 
                                           d=X.shape[1], 
                                           R=R[self.kernel]**self._d)
            
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))
    
    
    
    def _compute_normalization(self):
        if self.kernel == 'box':
            self._normalization = (2 * self._h)** self._d
            
        elif self.kernel == 'gaussian':
            self._normalization = (self._h * np.sqrt(2 * np.pi))** self._d
            
        elif self.kernel == 'epa':
            self._normalization = self._h** self._d * volume_unit_ball(self._d)
        
        self._normalization = self._n * self._normalization 
        
            
    def set_params(self, **params):
        """
        Set parameters.

        Parameters
        ----------
        **params : kwargs
            Parameters et values to set.

        Returns
        -------
        self : DensityEstimator
            The self object.

        """
        for param, value in params.items():
            setattr(self, param, value)

def volume_unit_ball(d, p=2):
    # from KDEpy
    return 2.0 ** d * gamma(1 + 1 / p) ** d / gamma(1 + d / p)

def scott_bandwidth(n, d):
    """
    Scott's rule according to "Multivariate density estimation", Scott 2015, p.164.
    The Silverman's rule is exactly the same in "Density estimation for statistics and data analysis", Silverman 1986, p.87.
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features).

    Returns
    -------
    h : float
        Scotts rule bandwidth. The returned bandwidth should be then factored
        by the data variance.

    """
    return(n**(-1/(d + 4)))

def terrel_bandwidth(n, d, R):
    num = (d + 8) ** ((d+6)/2) * np.pi ** (d / 2) * R
    den = 16 * n * gamma((d+8)/2) * (d + 2)
    return (num / den)**(1/(d+4))

def generate_grid(*kwargs):
    xx = np.meshgrid(*kwargs)
    
    xx = np.vstack([xi.flat for xi in xx]).T
    
    return(xx)