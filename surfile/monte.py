"""
Monte carlo simulation 
for uncertainty estimation
- analysis of roughness features for:
    - Profiles

@author: Andrea Giura
"""

import copy

import monaco as mc
from scipy.stats import uniform, norm
import numpy as np
import matplotlib.pyplot as plt

from surfile import profile, roughness, filter


class RoughnessMC:
    @staticmethod
    def simulate(p: profile.Profile, sigma, ndraws, singleThread=True, bplt=False):
        """
        Calculates roughness parameters varying the profile
        given a gaussian distribution for the point uncertainty
        Apllies a MonteCarlo approach to calculate the parameters
        distributions and evaluate parameter uncertainty.

        Parameters
        ----------
        p : profile.Profile
            The profile on wich the MonteCarlo analisys is performed
        sigma : float
            The uncertainty value used to modify the profile
        ndraws : int
            The number of different profiles tested
        singleThread : bool, optional
            If false the program uses monaco multithread
            routines to parallelize the computation, a localhost
            can be opened with the process statistics (bokeh required)
        bplt : bool, optional
            If true plots all the profile randomly generated
            during the simulation, if a multithreaded process
            is selected bplt is set to False to limit the number
            of lines plotted
        """
        def mc_pre(case):
            # d_m = case.invals['d mean [nm]'].val
            p = case.constvals['p']
            sigma = case.constvals['sigma']
            mu = case.constvals['mu']
            return p, mu, sigma

        def mc_run(p: profile.Profile, mu, sigma):
            """
            The routine for the parameter calculation
            - form removal
            - filter
            - param calc
            """
            disp = sigma * np.random.randn(p.X.size) + mu
            newp = copy.deepcopy(p)
            newp.Z += disp
            
            if bplt: bx.plot(newp.X, newp.Z, alpha=0.5)

            # if no filter is needed set fil=None
            fil = filter.ProfileGaussian(cutoff=1)
            RA, RQ, RP, RV, RT, RSK, RKU = roughness.Parameters.calc(newp, fil=fil)
            return RA, RQ, RP, RV, RT, RSK, RKU

        def mc_post(case, RA, RQ, RP, RV, RT, RSK, RKU):
            case.addOutVal('ra [um]', RA)
            case.addOutVal('rq [um]', RQ)
            case.addOutVal('rp [um]', RP)
            case.addOutVal('rv [um]', RV)
            case.addOutVal('rt [um]', RT)
            case.addOutVal('rsk [um]', RSK)
            case.addOutVal('rku [um]', RKU)

        if not singleThread: bplt=False
        seed = 78547876
        fcns = {'preprocess': mc_pre,
                'run': mc_run,
                'postprocess': mc_post}
        fcns = {'preprocess': mc_pre,
                'run': mc_run,
                'postprocess': mc_post}

        sim = mc.Sim(name='Roughness', ndraws=ndraws, fcns=fcns, firstcaseismedian=True,
                     seed=seed, singlethreaded=singleThread, verbose=True, debug=False,
                     savecasedata=False, savesimdata=False)

        sim.addConstVal(name='p', val=p)
        sim.addConstVal(name='sigma', val=sigma)
        sim.addConstVal(name='mu', val=0)
        
        if bplt:
            fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
            ax.plot(p.X, p.Z)

        sim.runSim()
        return sim.outvars.values()
    
