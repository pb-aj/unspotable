"""
File to calculate eigens following eigenmapping process.
Code sourced from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885

Modifications made by A.J. deVaux (https://github.com/pb-aj/un-spot-able)
"""

import numpy as np
import pca

def mkcurves(star, nt, lmax, ncurves):
    """
    Generates light curves for each spherical harmonic mode (excludin Y00) of passed star
    Runs the light curve design matrix (lcs) through pca to generate eigencurve design matrix
    and eigenmaps with observable and null components separated

    Arguments
    ---------
    star: object
        A starry star object, initialized with a cfg file

    nt: integer
        Length of each light curve array

    lmax: integer
        Maximum spherical harmonic degree (l) to use

    ncurves: integer
        Number of light curves to generate.  
        If left as None, it will equal nharm which is (lmax + 1)**2 - 1

    use_y00: boolean
        Determines if code should remove y00 when calculating light curves
        Based on value in configuration file

    Returns
    -------
    eigeny: 2D array
        (ncurves, ny) array of y coefficients for each harmonic. 
        ncurves is the number of harmonics, including only positive versions and excluding Y00: (lmax + 1)**2 - 1
        ny is the number of y coefficients to describe a harmonic with degree lmax: (lmax + 1)**2 (includes y00)

    evalues: 1D array
        (ncurves) array of eigenvalues sorted by variance

    evectors: 2D array
        (ncurves, nt) array of normalized (unit) eigenvectors

    proj: 2D array ("ecurves")
        (ncurves, nt) array of the data projected in the new space (the PCA "eigencurves") 
        Note, the imaginary part is discarded

    lcs: 2D array
        (ncurves, nt) array of the light curves that are passed into pca to generate the other return values
    """    

    #Set up theta array
    thet = np.linspace(0, 360, nt)

    #Pull out lightcurve design matrix & remove uniform contributions
    lcs = star.map.design_matrix(theta=thet).eval().T[1:,:]
            
    # Run PCA to create orthogonal eigencurve basis set
    evalues, evectors, proj = pca.pca(lcs, ncomp=ncurves)

    # Discard imaginary part of eigens to appease numpy
    proj = np.real(proj)
    evalues = np.real(evalues)
    evectors = np.real(evectors)

    # Utilize eigens to generate eigenmaps using spherical harmonic coefficients
    eigeny = np.zeros((ncurves, (lmax + 1)**2))
    eigeny[:,0] = 1.0 # Y00 = 1 for all maps

    for j in range(ncurves):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                # (ok because evectors has only been sorted along one dimension)
                eigeny[j,yi] = evectors.T[j,shi]
                yi  += 1
                shi += 1

    return eigeny, evalues, evectors, proj, lcs



            
