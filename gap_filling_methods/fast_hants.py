'''
Gapfilling approaches using a simplified implementation of HANTS
'''
try:    
    import numpy as np
    import numpy.linalg as la
    import gc, os

    class FastHANTS():    
        """
        Based on: Roerink et at. "Reconstructing cloudfree NDVI composites using Fourier analysis of time series".
        Optimization based on precomputation for different combinations of gaps pattern in a year
        
        :param ts: time seires of a pixel
        :param n_ipy: number of images per year
        :param n_freq: number of Fourier coefficients to get (in addition to the constant value)
        :param reg_param: regularization parameter for the pseudoinversion
        :param min_ipy_pct: minumum percentage of images per year to perform the gap-filling
        :param keep_orig_values: do not modify the values of the time series that are not gaps
        """
        
        def __init__(s,
                     ts:np.array,
                     n_ipy:int,
                     n_freq:int = 2,
                     reg_param:float = 0.05,
                     min_ipy_pct:float = 50,
                     n_cpu:int = os.cpu_count(),
                     keep_orig_values:bool = True,
                    ):
            assert (ts.shape[2]%n_ipy) == 0, "The number of imagies is supposed to be a multiple of the images per year"
            try:
                import mkl
                mkl.set_num_threads(n_cpu)
            except:
                pass
            s.orig_shape = ts.shape
            s.ts = np.reshape(ts, (ts.shape[0]*ts.shape[1], ts.shape[2])).copy()
            del ts
            gc.collect()
            n_y = np.floor(s.ts.shape[1]/n_ipy).astype(int)
            s.ts = np.reshape(s.ts, (s.ts.shape[0]*n_y, n_ipy))            
            s.keep_orig_values = keep_orig_values
            s.n_ipy = n_ipy
            s.n_ts = s.ts.shape[0]
            s.n_freq = n_freq
            s.reg_param = reg_param
            gaps_mask = np.isnan(s.ts)
            year_gaps = np.sum(gaps_mask.astype(int), axis=1)
            s.idx_to_fill = (year_gaps>0) & (year_gaps<n_ipy-np.ceil(min_ipy_pct/100*n_ipy).astype(int))
            s.ts_to_fill = s.ts[s.idx_to_fill,:]
            s.gaps_mask_to_fill = gaps_mask[s.idx_to_fill,:]
            s.ts_to_fill[s.gaps_mask_to_fill] = 0.0
            s._precompute_DFT_matrix()
            s._precompute_V_matrix()
            
        def _precompute_DFT_matrix(s):
            ## Creating the DFT matrix
            n_row = min(2*s.n_freq+1, s.n_ipy-1+s.n_ipy%2);
            s.F = np.zeros((n_row, s.n_ipy))
            s.F[0,:] = np.sqrt(s.n_ipy) # The scalings are necessay to get and unitary matrix
            for i in np.arange(np.floor(n_row/2).astype(int)):
                period = np.arange(s.n_ipy) / s.n_ipy*np.pi*2*(i+1)
                s.F[2*i+1,:] = np.cos(period) * np.sqrt(2/s.n_ipy)
                s.F[2*(i+1),:] = np.sin(period) * np.sqrt(2/s.n_ipy)
                
        def _precompute_V_matrix(s):
            # Getting unique ID for each gaps pattern    
            s.pattern_ids = np.sum(s.gaps_mask_to_fill*2**np.arange(s.n_ipy), axis=1)
            unique_ids = np.unique(s.pattern_ids)
            n_ui = unique_ids.shape[0]
            s.V_matrix = np.zeros((n_ui, s.F.shape[0], s.n_ipy))
            # Precomputing the pseudoinverses and the right side of the matrix skeleton for the unique gaps patterns
            for i, pattern_id in enumerate(unique_ids):
                idx = s.pattern_ids==pattern_id
                unique_mask = s.gaps_mask_to_fill[idx,:][0,:]
                Ft = s.F.copy()
                Ft[:,unique_mask] = 0
                s.V_matrix[i,:,:] = la.pinv(Ft @ Ft.T + s.reg_param * np.eye(Ft.shape[0])) @ Ft
                s.pattern_ids[idx] = i
                            
        def run(s):
            ## For each series of yearly images: perform a regularized LSE pseudoinversion of the DFT matrix to get the Fourier coeffiecents based on available data and project back to the time series
            ts_rec = np.einsum('jk,ij->ik', s.F, np.einsum('ijk,ik->ij', s.V_matrix[s.pattern_ids,:,:], s.ts_to_fill))
            if s.keep_orig_values:
                ts_rec[~s.gaps_mask_to_fill] = s.ts_to_fill[~s.gaps_mask_to_fill]
            s.ts[s.idx_to_fill,:] = ts_rec
            s.ts = np.reshape(s.ts, (s.orig_shape[0]*s.orig_shape[1], s.orig_shape[2]))                 
            return np.reshape(s.ts, s.orig_shape)
            
            
except ImportError as e:
    from .misc import _warn_deps
    _warn_deps(e, 'fast_hants')
