import copy
from abc import ABC, abstractmethod

import monaco as mc
from scipy.stats import uniform, norm
import numpy as np
import matplotlib.pyplot as plt

from surfile import profile, roughness, filter


class RoughnessMC:
    @staticmethod
    def simulate(p: profile.Profile, ndraws):
        def mc_pre(case):
            # d_m = case.invals['d mean [nm]'].val
            pr = case.constvals['p']
            sigma = case.constvals['sigma']
            mu = case.constvals['mu']
            return pr, mu, sigma
    
        def mc_run(pr: profile.Profile, mu, sigma):
            disp = sigma * np.random.randn(pr.X.size) + mu
            newp = copy.deepcopy(pr)
            newp.Z += disp

            bx.plot(newp.X, newp.Z, alpha=0.5)
    
            fil = filter.ProfileGaussian(1)
            RA, RQ, RP, RV, RT, RSK, RKU = roughness.Parameters.calc(newp, fil=fil)
            return RA, RQ, RP, RV, RT, RSK, RKU
    
        def mc_post(case, RA, RQ, RP, RV, RT, RSK, RKU):
            case.addOutVal('ra [nm]', RA)
            case.addOutVal('rq [nm]', RQ)
            case.addOutVal('rp [nm]', RP)
            case.addOutVal('rv [nm]', RV)
            case.addOutVal('rt [nm]', RT)
            case.addOutVal('rsk [nm]', RSK)
            case.addOutVal('rku [nm]', RKU)

        seed = 78547876
        fcns = {'preprocess': mc_pre,
                'run': mc_run,
                'postprocess': mc_post}

        sim = mc.Sim(name='Roughness', ndraws=ndraws, fcns=fcns, firstcaseismedian=True,
                     seed=seed, singlethreaded=True, verbose=True, debug=True,
                     savecasedata=False, savesimdata=False)

        sim.addConstVal(name='p', val=p)
        sim.addConstVal(name='sigma', val=0.04)
        sim.addConstVal(name='mu', val=0)
        
        fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        ax.plot(p.X, p.Z)

        sim.runSim()

        figh, axsh = plt.subplots(nrows=1, ncols=len(sim.outvars))
        for i, ovar in enumerate(sim.outvars.values()):
            mc.plot_hist(ovar, ax=axsh[i])
            print(ovar.stats())

        plt.show()
