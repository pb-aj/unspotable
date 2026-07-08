"""
File to run PCA to determine eigencurve basis set following eigenmapping process.  
Code sourced from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885

Modifications made by A.J. deVaux (https://github.com/pb-aj/un-spot-able)
"""

import numpy as np
from sklearn.decomposition import PCA

def pca(arr, ncomp=None):
    """
    Runs principle component analysis on the input array using sci-kit learn PCA 
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    Pedregosa, F., Varoquaux, G., Gramfort, A., et al. 2011, Journal of Machine Learning Research, 12, 2825

    Arguments
    ---------
    arr: 2D array
        (m, n) array, where n is the size of the dataset (e.g., times
        in an observation) and m is the number of vectors

    ncomp: int (optional)
        Number of component to keep as defined by sci-kit learn.  
        Default to None which keeps all components; code will set to number of harmonic modes (m)

    Returns
    -------
    evalues: 1D array
        array of eigenvalues of size n

    evectors: 2D array
        (m, m) array of sorted eigenvectors

    proj: 2D array
        (m, n) array of data projected in the new space

    """

    #Run PCA using ncomp and svd_solver = "full"
    pca = PCA(n_components=ncomp, svd_solver = "full")
    proj = pca.fit_transform(arr.T).T

    #Extract needed eigen information from PCA fit
    evalues  = pca.explained_variance_
    evectors = pca.components_
    evectors = evectors.T
            
        
    return evalues, evectors, proj
