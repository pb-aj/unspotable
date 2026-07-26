"""
Main file to run un-spot-able code
https://github.com/pb-aj/un-spot-able
"""

#general imports
import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
import starry2 as starry
import faulthandler
faulthandler.enable()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')

# Lib imports
sys.path.append(libdir)
from lib.spotable import spotable as se

# py imports
import create_eigens
import create_emaps
import create_rv
import create_real_null

# Set up starry configuration
starry.config.quiet = True
starry.config.lazy = True
sys.setrecursionlimit(10000) # starry seems to have a lot of recursion


# Whether or not to show print statements in code
dpm = True #change to False to make quiet

def unspotable(cfile):

    se("\n\033[32mCalling create_eigens:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    eigeny, evalues, evectors, ecurves, lcs, star, fit = \
        create_eigens.create_eigens(cfile,prompt_user=False)
    
    emaps_path = create_emaps.set_emap_directory(fit)

    se("\n\033[32mCreating emap visualizations:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)


    create_emaps.create_emaps(star, eigeny, emaps_path=emaps_path)

    se("----------------------------------------------------------------------------", dp = dpm)

    se("\n\033[32mCreating light curve visualizations:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    create_emaps.create_eflux(star, eigeny, emaps_path=emaps_path)

    se("----------------------------------------------------------------------------", dp = dpm)

    se("\n\033[32mNormailizing null eigen results:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)
    null_eigens, fit = create_real_null.normalize_null_eigens(cfile, prompt_user=False)

    results_path = create_real_null.set_real_directory(fit)

    se("\n\033[32mCreating realistic null maps:\033[0m", dp = dpm)
    se("----------------------------------------------------------------------------", dp = dpm)

    if fit.cfg.star.units:
        scaler = 5.670374419e-8 * fit.cfg.star.teff**4
        power = int(np.floor(np.log10(scaler)))
        if power > 6:
            units = f"W m$^{{-2}}$ * 10$^{power}$"
        else:
            units = f"W m$^{{-2}}$"
    else:
        scaler = 1
        units = None

    create_real_null.create_real_null(null_eigens, fit, results_path, units=units, scale_star=scaler)

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

    # Call unspotable
    unspotable(cfile)

    sys.exit("\033[32mdone\033[0m")