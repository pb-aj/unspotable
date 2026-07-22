"""
File to convert null eigen maps to have realistic flux ranges and then calculate results
https://github.com/pb-aj/un-spot-able
"""

#general imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import starry2 as starry
from cmcrameri import cm
import faulthandler
faulthandler.enable()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
from lib import utils
from lib import fitclass    as fc
from lib.spotable import spotable as se

# py imports
import create_eigens
import create_emaps
import create_rv

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion


# Whether or not to show print statements in code
dpm = True #change to False to make quiet

def normalize_null_eigens(cfile, prompt_user=True):

    fit = fc.Fit()

    se("\tReading the configuration file & data",dp = dpm)

    try:
        fit.read_config(cfile)
    except:
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit('\033[31mERROR:\033[0m' + ' Check Name of Configuration File.',dp = dpm)
    
    cfg = fit.cfg

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    norm_eigen_path = os.path.join(outdir,"stored-norm-null-eigens")

    se(f"\n\tSetting directory to:\n\t\033[34m{outdir}\033[0m\n",dp = dpm)

    if prompt_user:
        if os.path.exists(norm_eigen_path):
            use_stored_eigeny = input("\tWould you like to use the stored normalized null eigen results? (y/n) ")
        else:
            se("\tNo stored normalized null eigen results found, creating new ones...",dp = dpm)
            os.mkdir(norm_eigen_path)
            use_stored_eigeny = "n"
    else:
        if os.path.exists(norm_eigen_path):
            use_stored_eigeny = "y"
        else:
            se("\tNo stored normalized null eigen results found, creating new ones...",dp = dpm)
            os.mkdir(norm_eigen_path)
            use_stored_eigeny = "n"

    if use_stored_eigeny.lower().strip() == "y":
        if prompt_user:
            se("",dp=dpm)
            
        se(f"\tReading previously stored normalized null eigen results from:\n\t\033[34m{norm_eigen_path}\033[0m",dp = dpm)
        try:
            norm_eigeny = np.loadtxt(f"{norm_eigen_path}/norm_null_eigeny.txt")
            se("----------------------------------------------------------------------------", dp = dpm)

            return norm_eigeny, fit
        except:
            se(f"\n\t\033[31m\033[1mStored normalized null eigen results are invalid, calculating new ones...\033[0m",dp = dpm)
            use_stored_eigeny = "n"
    elif prompt_user:
        se("\n\tGenerating new normalized null eigen results...",dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    se("\n\033[32mCalling create_eigens to create initial maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    eigeny, evalues, evectors, ecurves, lcs, star, fit = \
        create_eigens.create_eigens(cfile,prompt_user=prompt_user)
    

    se("\n\033[32mNormalizing Null Maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)   

    se("\tCreating a new star with rv=True", dp = dpm)
    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg

    rv_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)

    se("\n\tFinding rank of the rv star design matrix:", dp = dpm)

    A = star.map.design_matrix(theta=np.linspace(0,360,100)).eval()
    R = np.empty((1, lmax))

    R = [
        np.linalg.matrix_rank(A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    ]

    #Display rank for each spherical harmonic degree
    se("\t------------------------------------------------",dp = dpm)
    se(f"\t\033[1mThe rank (# non-null maps) for lmax = {lmax} (ncurves = {int(A.shape[1])}) is {int(R[-1])}\033[0m",dp = dpm) 
    se("\t------------------------------------------------\n",dp = dpm)

    se("\tUsing rank to pull out only null space maps", dp = dpm)
    null_eigeny = eigeny[int(R[-1] - 1):]

    se("\n\tCentering null maps at 0 Flux", dp = dpm)
    for i in range(null_eigeny.shape[0]):

        rv_star.map[:,:] = null_eigeny[i]
        rv_star.map[0,0] = 0

        correction_f = rv_star.map.flux(theta=np.linspace(0,360,60)).eval()
        null_eigeny[i,0] = -np.median(correction_f)


    se(f"\n\tStoring normalized null maps to:\n\t\033[34m{norm_eigen_path}\033[0m", dp = dpm)
    np.savetxt(f"{norm_eigen_path}/norm_null_eigeny.txt", null_eigeny)

    se("----------------------------------------------------------------------------", dp = dpm)   

    return null_eigeny, fit


def set_real_directory(fit):
    """
    
    """

    cfg = fit.cfg

    subdir = cfg.folder
    outdir = os.path.join(cfg.outdir, subdir)
    results_path = os.path.join(outdir,"realistic_outputs")

    se(f"\tSetting directory to:\n\t\033[34m{results_path}\033[0m\n", dp = dpm)

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    return results_path


def set_up_stars(fit, eigeny, uni_comp = 1, theta_neg_90 = np.linspace(-90,90,180)):

    cfg = fit.cfg
    lmax = cfg.sim.lmax
    udeg = cfg.star.udeg

    #load in data/set up stars
    realistic_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    inv_realistic_star = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)

    uni_star_limb = utils.initstar(fit, lmax, udeg=udeg, include_rv=True)
    uni_star = utils.initstar(fit, lmax, include_rv=True)

    #reset the star, create version without limb darkening, and create versions of unfirom stars
    realistic_star.map[0,0] = eigeny[0] + uni_comp
    realistic_star.map[1:,:] = eigeny[1:]

    inv_realistic_star.map[0,0] = -eigeny[0] + uni_comp
    inv_realistic_star.map[1:,:] = -eigeny[1:]

    #central intenisty is the same regardless of inc, so setting to 90 for ease
    uni_star.map.inc = 90
    uni_star_limb.map.inc = 90

    uni_star.map[0,0] = uni_comp
    uni_star.map[1:,:] = 0

    uni_star_limb.map[0,0] = uni_comp
    uni_star_limb.map[1:,:] = 0

    uni_I = uni_star.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval()
    uni_I_limb = uni_star_limb.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval()

    ratio_no_limb = np.max(uni_I) / np.max(uni_I_limb)

    return realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb



def set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb, 
                    uni_comp = 1, theta_neg_90 = np.linspace(-90,90,180),
                    lower_fraction = .42, upper_fraction = 1.35, interval=.1):
    
    uni_star_limb.map.amp = ratio_no_limb
    

    uni_star_limb.map[0,0] = uni_comp #* ratio_no_limb
    uni_star_limb.map[1:,:] = 0
    uni_value = np.max(uni_star_limb.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval())


    amount_below_uni = uni_value - realistic_star.map.minimize()[-1].eval()
    amount_above_uni = uni_value - inv_realistic_star.map.minimize()[-1].eval()

    ratio_below = (uni_value - amount_below_uni) / uni_value
    ratio_above = (amount_above_uni + uni_value) / uni_value
    

    while ratio_below <= lower_fraction or ratio_above >= upper_fraction:

        """
        Note: starry does not apply limb darkening to render or minimize.  
        So while these maps need to be scaled for .show() animations and intensity curves, 
        they do not need to be scaled when dealing with static plots/minimize/anything that
        can't have limb darkening applied
        """

        uni_comp += interval

        uni_star_limb.map[0,0] += interval
        uni_value = np.max(uni_star_limb.map.intensity(lat=0, lon=theta_neg_90,rv=False).eval())

        realistic_star.map[0,0] += interval

        inv_realistic_star.map[0,0] += interval

        amount_below_uni = uni_value - realistic_star.map.minimize()[-1].eval()
        amount_above_uni = uni_value - inv_realistic_star.map.minimize()[-1].eval()

        ratio_below = (uni_value - amount_below_uni) / uni_value
        ratio_above = (amount_above_uni + uni_value) / uni_value

    """
    As described in the note above, we want to apply the amp after because limb only impacts intensity and animated plots
    Therefore if we apply it before we will have the same issue where the limb-calcs will be inflated unrealistically
    This means we also need to turn the amp "off" for any static plots/calc without limb involved.
    """

    # Applying the ratio_no_limb to Star's Amp to fix Starry limb darkening.
    realistic_star.map.amp = ratio_no_limb


    return realistic_star, inv_realistic_star, uni_star_limb, uni_comp, ratio_below, ratio_above, uni_value

def scale_real_map(realistic_star, uni_star_limb, scale_value, uni_comp, ratio_no_limb, scaler="lc"):

    realistic_star.map.amp = ratio_no_limb
    uni_star_limb.map.amp = ratio_no_limb

    if scaler == "lc":

        scaled_amp = scale_value / uni_comp / ratio_no_limb

        realistic_star.map.amp*= scaled_amp

        uni_star_limb.map.amp*= scaled_amp

    elif scaler == "intensity":
        uni_intensity = np.max(uni_star_limb.map.intensity(lat=0, lon=np.linspace(-90,90,180),rv=False).eval())

        factor = scale_value / uni_intensity

        realistic_star.map.amp *= factor
        
        uni_star_limb.map.amp *= factor

    
def create_real_null(null_eigens, fit, results_path, theta=np.linspace(0,360,180), uni_comp = 1,
                     scale_star=1, scaler="lc", cmap=cm.buda, store_scaled=True):

    se("\tCalculating Flux Range Limits\n", dp=dpm)

    cfg = fit.cfg

    ani_interval = int(6000 / len(theta))

    teff = cfg.star.teff
    max_teff = cfg.star.max_teff
    min_teff = cfg.star.min_teff

    min_ratio = np.round(min_teff**4 / teff**4, decimals=2)

    max_ratio = np.round(max_teff**4 / teff**4, decimals=2)

    if store_scaled:
        scaled_null_eigens = []

    se("\tLooping through each null map and generating plots:",dp = dpm)
    se("\t------------------------------------------------",dp = dpm)

    for i in range(null_eigens.shape[0]):

        folder_name = f"null_map_{i}"

        if not os.path.isdir(f"{results_path}/{folder_name}"):
            os.mkdir(f"{results_path}/{folder_name}")

        se("\t\t\u2022 Setting up reference stars",dp = dpm)

        realistic_star, inv_realistic_star, uni_star_limb, uni_star, ratio_no_limb = set_up_stars(fit, null_eigens[i], uni_comp=uni_comp)

        se("\t\t\u2022 Adjusting star to realistic range",dp = dpm)

        realistic_star, inv_realistic_star, uni_star_limb, adj_uni_comp, ratio_below, ratio_above, uni_value = \
            set_up_real_map(realistic_star, inv_realistic_star, uni_star_limb, ratio_no_limb,
                            lower_fraction=min_ratio,upper_fraction=max_ratio, uni_comp=uni_comp,
                            interval=.2)

        se(f'\t\t\u2022 Range of flux for "Null map {i}" is now {ratio_below:.2f} to {ratio_above:.2f}', dp=dpm)
        
        if not scale_star is None:
            if scaler == "lc":
                se(f"\t\t\u2022 Scaling star lightcurve to {scale_star}",dp = dpm)
            else:
                se(f"\t\t\u2022 Scaling star uniform intensity to {scale_star}",dp = dpm)

            scale_real_map(realistic_star, uni_star_limb, 
                        scale_value=scale_star, uni_comp=adj_uni_comp, 
                        ratio_no_limb=ratio_no_limb,scaler=scaler)
            
        if store_scaled:
            scaled_null_eigens.append(realistic_star.map.y.eval())
            

        se(f'\t\t\u2022 Creating plots for "Null map {i}"', dp=dpm)
        
        create_rv.map_animations(realistic_star, theta = theta,fname=f"{results_path}/{folder_name}/emap_animation.mp4", 
                       interval = ani_interval, transparent=False, cmap=cmap)
        
        create_rv.map_animations(realistic_star, theta = theta[::2],fname=f"{results_path}/{folder_name}/emap_animation.gif", 
                       interval = ani_interval, transparent=False, cmap=cmap)

        create_rv.flux_rv_line(realistic_star,
                     flux_name=f"{results_path}/{folder_name}/flux_curve.png",rv_name=f"{results_path}/{folder_name}/rv_curve.png")
        
        realistic_star.map.amp /= ratio_no_limb
        create_emaps.emap_plot(realistic_star, indiv_path=f"{results_path}/{folder_name}", proj='rect', other_fname=None, 
                 transparent=False, colorbar=True, cmap=cmap)
        
        create_emaps.emap_plot(realistic_star, indiv_path=f"{results_path}/{folder_name}", proj='moll', other_fname=None, 
                 transparent=False, colorbar=True, cmap=cmap)
        realistic_star.map.amp *= ratio_no_limb

        se(f'\033[38;5;208m\t\t"Null map {i}" plots are complete!\033[0m', dp=dpm)
        se("\t\t------------------------------------------------", dp=dpm)
    
    if store_scaled:

        subdir = cfg.folder
        outdir = os.path.join(cfg.outdir, subdir)
        norm_eigen_path = os.path.join(outdir,"stored-norm-null-eigens")
        np.savetxt(f"{norm_eigen_path}/scaled-null-eigeny.txt", np.array(scaled_null_eigens))


if __name__ == "__main__":
    # Uncomment if you want to see command line arguments 
    # se(sys.argv,dp = dpm) 

    # Check command line input is correct
    if len(sys.argv) < 2:
        se("\n----------------------------------------------------------------------------",dp = dpm)
        se('\033[31mERROR:\033[0m' + ' Call structure is "\033[34mpython create_real_null.py <configuration file>\033[0m"',dp = dpm)
        se("----------------------------------------------------------------------------",dp = dpm)
        sys.exit()
    else:
        cfile = sys.argv[1]

    se("\n\033[32mNormailizing null eigen results:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    null_eigens, fit = normalize_null_eigens(cfile, prompt_user=False)

    se("\n\033[32mCreating realistic null maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    results_path = set_real_directory(fit)

    create_real_null(null_eigens, fit, results_path)

    sys.exit("\033[32mdone\033[0m")

