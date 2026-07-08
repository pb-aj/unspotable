"""
File to create eigens, check their rank, and save them for a stellar system 
https://github.com/pb-aj/un-spot-able
"""


# General imports
import os
import sys
import numpy as np
import starry2 as starry
import faulthandler
faulthandler.enable()

# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# lib imports
sys.path.append(libdir)
from lib import eigen
from lib import utils
from lib import fitclass as fc
from lib.spotable import spotable as se

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion

# Whether or not to show print statements in code
dpm = True #change to False to make quiet

def create_eigens(cfile, prompt_user=True):
    """
    Calculates all eigens according to passed cfile and saves them to txt files.  
    If saved eigens already exist, gives option to read them instead of re-calculating.

    Arguments
    ---------
    cfile: string
        name of configuration file

    prompt_user: boolean (optional)
        Whether or not to prompt the user to use stored eigens.  When True (default), will ask user.
        When False, will use stored eigens if possible without prompting.

    Returns
    -------
    eigeny: 2D array
        (ncurves, ny) array of y coefficients for each harmonic. 
        ncurves is the number of harmonics, including only positive versions and excluding Y00: (lmax + 1)**2 - 1
        ny is the number of y coefficients to describe a harmonic with degree lmax: (lmax + 1)**2 (includes y00)
    evalues: 1D array
        (ncurves) array of eigenvalues sorted by variance
    evectors: 2D array
        (ncurves, nlcs) array of normalized (unit) eigenvectors
    ecurves: 2D array ("proj")
        (ncurves, nt) array of the data projected in the new space (the PCA "eigencurves") 
        Note, the imaginary part is discarded
    lcs: 2D array
        (ncurves, nt) array of the light curves that are passed into pca to generate the other return values
    star: object
        A starry star object, initialized with a cfg file
    fit: fit class (defined in lib/fitclass.py)
        stored fit values read from cfg
    """    

    se("\nSetting-Up:",dp = dpm)
    se("----------------------------------------------------------------------------",dp = dpm)

    fit = fc.Fit()

    se("\tReading the configuration file & data",dp = dpm)

    try:
        fit.read_config(cfile)
    except:
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit('\033[31mERROR:\033[0m' + ' Check Name of Configuration File.',dp = dpm)

    cfg = fit.cfg

    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg

    star = utils.initstar(fit, lmax, udeg=udeg)

    se("\n\tFinding rank of the spherical harmponic design matrix:",dp = dpm)
    se("\t------------------------------------------------",dp = dpm)

    A = star.map.design_matrix(theta=np.linspace(0,360,181)).eval()

    R = np.linalg.matrix_rank(A)

    #Display rank for each spherical harmonic degree
    se(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(A.shape[1])}) is {int(R)}\033[0m",dp = dpm) 
    se("\t------------------------------------------------\n",dp = dpm)

    lmax = cfg.sim.lmax
    ncurves = int((lmax + 1) ** 2 - 1)
    nlcs = cfg.sim.nlcs

    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    se(f"\tSetting directory to:\n\t\033[34m{outdir}\033[0m",dp = dpm)

    if not os.path.isdir(os.path.join(cfg.outdir, subdir)):
        os.mkdir(os.path.join(cfg.outdir, subdir))


    se("\n\tCreating star with parameters from cfg file",dp = dpm)
    star = utils.initstar(fit, lmax, udeg=udeg)

    if not star.map.limbdark_is_physical().eval():
        se("----------------------------------------------------------------------------",dp = dpm)
        se("\033[31mWARNING:\033[0m Limb Darkening is not physical!",dp = dpm)


    eigen_path = os.path.join(outdir,"stored-eigens")
    if prompt_user:
        if os.path.exists(eigen_path):
            se("----------------------------------------------------------------------------\n",dp = dpm)
            use_stored_pca = input("Would you like to use the stored eigen results? (y/n) ")
        else:
            se("----------------------------------------------------------------------------\n",dp = dpm)
            se("No stored eigen results found, creating new ones:",dp = dpm)
            os.mkdir(eigen_path)
            use_stored_pca = "n"
    else:
        if os.path.exists(eigen_path):
            se("----------------------------------------------------------------------------\n",dp = dpm)
            se("Attempting to read in existing eigen results:",dp = dpm)
            use_stored_pca = "y"
        else:
            se("----------------------------------------------------------------------------\n",dp = dpm)
            se("No stored eigen results found, creating new ones:",dp = dpm)
            os.mkdir(eigen_path)
            use_stored_pca = "n"


    if use_stored_pca.lower().strip() == "y":
        se("----------------------------------------------------------------------------",dp = dpm)
        se(f"\tReading previously stored eigen results from:\n\t\033[34m{eigen_path}\033[0m",dp = dpm)
        try:
            eigeny = np.loadtxt(f"{eigen_path}/eigeny.txt")
            evalues = np.loadtxt(f"{eigen_path}/evalues.txt")
            evectors = np.loadtxt(f"{eigen_path}/evectors.txt")
            ecurves = np.loadtxt(f"{eigen_path}/ecurves.txt")
            lcs = np.loadtxt(f"{eigen_path}/lcs.txt")
        except:
            se(f"\n\033[31m\033[1mStored eigen results are invalid, calculating new ones:\033[0m",dp = dpm)
            use_stored_pca = "n"

    if use_stored_pca.lower().strip() != "y":
        se("----------------------------------------------------------------------------",dp = dpm)
        se(f"\tCalculating new eigen results and storing them in:\n\t\033[34m{eigen_path}\033[0m",dp = dpm)
        eigeny, evalues, evectors, ecurves, lcs = \
        eigen.mkcurves(star, nlcs, lmax, ncurves)
        
        np.savetxt(f"{eigen_path}/eigeny.txt", eigeny)
        np.savetxt(f"{eigen_path}/evalues.txt", evalues)
        np.savetxt(f"{eigen_path}/evectors.txt", evectors)
        np.savetxt(f"{eigen_path}/ecurves.txt", ecurves)
        np.savetxt(f"{eigen_path}/lcs.txt",lcs)


    se("----------------------------------------------------------------------------",dp = dpm)
    se("\nFinding rank of the ecurve design matrix:",dp = dpm)
    se("----------------------------------------------------------------------------",dp = dpm)
    ecurve_A = ecurves.T

    ecurve_R = np.linalg.matrix_rank(ecurve_A)

    #Display rank for each spherical harmonic degree of the ecurve design matrix
    se(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(ecurve_A.shape[1])}) is {int(ecurve_R)}\033[0m",dp = dpm)
    se("\n\tNote, the ecurve design matrix does not include the uniform map\n\tand thus the rank should one less than the spherical harmonic result.",dp = dpm)  
    se("----------------------------------------------------------------------------",dp = dpm)
    return eigeny, evalues, evectors, ecurves, lcs, star, fit



if __name__ == "__main__":
    # Uncomment if you want to see command line arguments 
    # se(sys.argv,dp = dpm) 

    # Check command line input is correct
    if len(sys.argv) < 2:
        se("\n----------------------------------------------------------------------------",dp = dpm)
        se('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_eigens.py <configuration file>\033[0m"',dp = dpm)
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit()
    else:
        cfile = sys.argv[1]

    # Call create_eigens function
    create_eigens(cfile)

    sys.exit("\033[32mdone\033[0m")


