"""
File to read in configuration file.
Code sourced from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885

Modifications made by A.J. deVaux (https://github.com/pb-aj/un-spot-able)
"""

import configparser as cp
import configclass as cc

class Fit:
    """
    A class to hold attributes and methods related to provided configuration (cfg) file.
    """
    def read_config(self, cfile):
        """
        Read a configuration file and set up attributes accordingly.

        Note that self.cfg is a Configuration instance, and self.cfg.cfg
        is a raw ConfigParser instance. The ConfigParser instance should
        be parsed into attributes of the Configuration() instance for
        simpler access within other routines that use the Fit class.
        """
        config = cp.ConfigParser()
        config.read(cfile)
        self.cfg = cc.Configuration()

        self.cfg.cfile = cfile
        self.cfg.cfg   = config

        # General options
        self.cfg.outdir     = self.cfg.cfg.get('General', 'outdir')
        self.cfg.folder     = self.cfg.cfg.get('General', 'folder')

        #Simulation Parameters
        self.cfg.sim.lmax = self.cfg.cfg.getint('Sim', 'lmax')
        self.cfg.sim.nlcs = self.cfg.cfg.getint('Sim', 'nlcs')

        #Stellar Parameters
        self.cfg.star.r    = self.cfg.cfg.getfloat('Star', 'r')
        self.cfg.star.prot = self.cfg.cfg.getfloat('Star', 'prot')
        self.cfg.star.inc = self.cfg.cfg.getfloat('Star', 'inc')
        self.cfg.star.obl = self.cfg.cfg.getfloat('Star', 'obl')
        self.cfg.star.udeg = [float(item.strip()) for item in self.cfg.cfg.get('Star', 'udeg').split(",")]
        self.cfg.star.veq = self.cfg.cfg.getfloat('Star', 'veq')
        self.cfg.star.teff = self.cfg.cfg.getfloat('Star', 'teff')
        self.cfg.star.max_teff = self.cfg.cfg.getfloat('Star', 'max_teff')
        self.cfg.star.min_teff = self.cfg.cfg.getfloat('Star', 'min_teff')
        self.cfg.star.units = self.cfg.cfg.getboolean('Star', 'units')
