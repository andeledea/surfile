import copy
from abc import ABC, abstractmethod

import monaco as mc
from scipy.stats import uniform, norm
import numpy as np
import matplotlib.pyplot as plt

from surfile import profile, roughness, filter


class RoughnessMC:
    @staticmethod
    def mc_pre(case):
        # d_m = case.invals['d mean [nm]'].val
        p = case.constvals['p']
        sigma = case.constvals['sigma']
        mu = case.constvals['mu']
        return p, mu, sigma

    @staticmethod
    def mc_run(p: profile.Profile, mu, sigma):
        disp = sigma * np.random.randn(p.X.size) + mu
        newp = copy.deepcopy(p)
        newp.Z += disp

        fil = filter.ProfileGaussian(1)
        RA, RQ, RP, RV, RT, RSK, RKU = roughness.Parameters.calc(newp, fil=fil)
        return RA, RQ, RP, RV, RT, RSK, RKU

    @staticmethod
    def mc_post(case, RA, RQ, RP, RV, RT, RSK, RKU):
        case.addOutVal('ra [nm]', RA)
        case.addOutVal('rq [nm]', RQ)
        case.addOutVal('rp [nm]', RP)
        case.addOutVal('rv [nm]', RV)
        case.addOutVal('rt [nm]', RT)
        case.addOutVal('rsk [nm]', RSK)
        case.addOutVal('rku [nm]', RKU)

    @staticmethod
    def simulate(p: profile.Profile, ndraws):
        seed = 78547876
        fcns = {'preprocess': RoughnessMC.mc_pre,
                'run': RoughnessMC.mc_run,
                'postprocess': RoughnessMC.mc_post}

        sim = mc.Sim(name='TMV', ndraws=ndraws, fcns=fcns, firstcaseismedian=True,
                     seed=seed, singlethreaded=True, verbose=True, debug=True,
                     savecasedata=False, savesimdata=False)

        sim.addConstVal(name='p', val=p)
        sim.addConstVal(name='sigma', val=0.02)
        sim.addConstVal(name='mu', val=0)

        sim.runSim()

        for ovar in sim.outvars.values():
            mc.plot_hist(ovar)
            print(ovar.stats())
        # sim.plot()

        plt.show()
