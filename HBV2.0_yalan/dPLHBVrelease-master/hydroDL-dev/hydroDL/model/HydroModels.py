import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
from . import cnn
import csv
import numpy as np
import random


class SACSMA(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(SACSMA, self).__init__()

    def forward(self, x, parameters, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
                comprout=False, dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS = 1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = SACSMA()
                buffpara = parameters[bufftime-1, :, :, :].unsqueeze(0).repeat(bufftime,1,1,1)
                Qsinit, S1,S2,S3,S4,S5 = initmodel(xinit, buffpara,staind,tdlst, mu, muwts, rtwts,
                                                                      bufftime=0,outstate = True, routOpt=False, comprout=False,dydrop = dydrop)

        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            S1 = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            S2 = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            S3 = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            S4 = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            S5 = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P_ = x[bufftime:, :, 0]
        Pm= P_.unsqueeze(2).repeat(1,1,mu) # precip
        T_ = x[bufftime:, :, 1]
        Tm = T_.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET
        parAll = parameters[bufftime:, :, :, :]
        parAllTrans = torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst = [[0,1], [1,2000], [0.005,0.995], [0.005,0.995], [0,1], [0,7],
                        [0.005,0.995], [0.005,0.995], [0,1], [0,1], [0,1]]  # HBV para
        routscaLst = [[0,2.9], [0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:] = hbvscaLst[ip][0] + parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])


        Nstep, Ngrid = P_.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static para matrix
        parhbvFull = torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat = torch.ones([1, Ngrid, 1])*dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix-1, :]
            dynPar = parAllTrans[:, :, ix-1, :]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar = dynPar*(1-drmask) + staPar*drmask
            parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables to save results
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()


        for t in range(Nstep):
            # paraLst = []
            # for ip in range(len(hbvscaLst)):  # unpack HBV parameters
            #     paraLst.append(parhbvFull[t, :, ip, :])

            # Fraction impervious area [-]
            pctim = parhbvFull[t, :, 0, :]
            # Maximum total storage depth [mm]
            smax = parhbvFull[t, :, 1, :]
            # fraction of smax that is Maximum upper zone tension water storage (uztwm) [-]
            f1 = parhbvFull[t, :, 2, :]
            # fraction of smax-uztwm that is Maximum upper zone free water storage (uzfwm) [-]
            f2 = parhbvFull[t, :, 3, :]
            # Interflow runoff coefficient [d-1]
            kuz = parhbvFull[t, :, 4, :]
            # Base percolation rate non-linearity factor [-]
            rexp = parhbvFull[t, :, 5, :]
            # fraction of smax-uztwm-uzfwm that is Maximum lower zone tension water storage (lztwm) [-]
            f3 = parhbvFull[t, :, 6, :]
            # fraction of smax-uztwm-uzfwm-lztwm that is Maximum lower zone primary free water storage (lzfwpm) [-]
            f4 = parhbvFull[t, :, 7, :]
            # Fraction of percolation directed to free water stores [-]
            pfree = parhbvFull[t, :, 8, :]
            # Primary baseflow runoff coefficient [d-1]
            klzp = parhbvFull[t, :, 9, :]
            # Supplemental baseflow runoff coefficient [d-1]
            klzs = parhbvFull[t, :, 10, :]

            uztwm = f1 * smax  # Maximum upper zone tension water storage [mm]
            uzfwm = torch.max(torch.tensor(0.005 / 4).cuda(),
                              f2 * (smax - uztwm))  # Maximum upper zone free water storage [mm]
            lztwm = torch.max(torch.tensor(0.005 / 4).cuda(),
                              f3 * (smax - uztwm - uzfwm))  # Maximum lower zone tension water storage [mm]
            lzfwpm = torch.max(torch.tensor(0.005 / 4).cuda(), f4 * (
                        smax - uztwm - uzfwm - lztwm))  # Maximum lower zone primary free water storage [mm]
            lzfwsm = torch.max(torch.tensor(0.005 / 4).cuda(), (torch.tensor(1.0).cuda() - f4) * (
                        smax - uztwm - uzfwm - lztwm))  # Maximum lower zone supplemental free water storage [mm]
            pbase = lzfwpm * klzp + lzfwsm * klzs  # Base percolation rate [mm/d]

            # Base percolation rate multiplication factor [-]: can return Inf, hence the min(10000,...)
            zperc = torch.min(torch.tensor(100000.0).cuda(),
                              (lztwm + lzfwsm * (torch.tensor(1.0).cuda() - klzs)) / (lzfwsm * klzs + lzfwpm * klzp) +
                              (lzfwpm * (torch.tensor(1.0).cuda() - klzp)) / (lzfwsm * klzs + lzfwpm * klzp))

            # Separate precipitation into liquid and solid components
            P = Pm[t, :, :]
            Ep = ETpm[t, :, :]
            delta_t = torch.tensor(1.0).cuda()
            flux_qdir = self.split_1(pctim, P)
            flux_peff = self.split_1(torch.tensor(1.0).cuda() - pctim, P)
            flux_ru = self.soilmoisture_1(S1, uztwm, S2, uzfwm)
            flux_euztw = self.evap_7(S1, uztwm, Ep, delta_t)
            flux_twexu = self.saturation_1(flux_peff, S1, uztwm)
            flux_qsur = self.saturation_1(flux_twexu, S2, uzfwm)
            flux_qint = self.interflow_5(kuz, S2)
            flux_euzfw = self.evap_1(S2, torch.max(torch.tensor(0.0).cuda(), Ep - flux_euztw), delta_t)
            flux_pc = self.percolation_4(pbase, zperc, rexp,
                            torch.max(torch.tensor(1e-8).cuda(), lztwm - S3) + torch.max(torch.tensor(1e-8).cuda(),  lzfwpm - S4)
                                         + torch.max( torch.tensor(1e-8).cuda(), lzfwsm - S5), torch.max(torch.tensor(1e-8).cuda(),lztwm + lzfwpm + lzfwsm), S2, uzfwm,
                                            delta_t)
            flux_pctw = self.split_1(torch.tensor(1.0).cuda() - pfree, flux_pc)
            flux_elztw = self.evap_7(S3, lztwm, torch.max(torch.tensor(0.0).cuda(), Ep - flux_euztw - flux_euzfw), delta_t)
            flux_twexl = self.saturation_1(flux_pctw, S3, lztwm)
            flux_twexlp = self.split_1(self.deficitBasedDistribution_pytorch(S4, lzfwpm, S5, lzfwsm), flux_twexl)
            flux_twexls = self.split_1(self.deficitBasedDistribution_pytorch(S5, lzfwsm, S4, lzfwpm), flux_twexl)
            flux_pcfwp = self.split_1(pfree * self.deficitBasedDistribution_pytorch(S4, lzfwpm, S5, lzfwsm), flux_pc)
            flux_pcfws = self.split_1(pfree * self.deficitBasedDistribution_pytorch(S5, lzfwsm, S4, lzfwpm), flux_pc)
            flux_rlp = self.soilmoisture_2(S3, lztwm, S4, lzfwpm, S5, lzfwsm)
            flux_rls = self.soilmoisture_2(S3, lztwm, S5, lzfwsm, S4, lzfwpm)
            flux_qbfp = self.baseflow_1(klzp, S4)
            flux_qbfs = self.baseflow_1(klzs, S5)

            S1 = torch.clamp( S1 + flux_peff   + flux_ru    - flux_euztw - flux_twexu,min=0.0001);
            S2 = torch.clamp(S2 + flux_twexu  - flux_euzfw - flux_qsur  - flux_qint  - flux_ru - flux_pc,min=0.0001);
            S3 = torch.clamp(S3 + flux_pctw   + flux_rlp   + flux_rls   - flux_elztw - flux_twexl,min=0.0001);
            S4 = torch.clamp(S4 + flux_twexlp + flux_pcfwp - flux_rlp   - flux_qbfp,min=0.0001);
            S5 = torch.clamp(S5 + flux_twexls + flux_pcfws - flux_rls   - flux_qbfs,min=0.0001);


            Qsimmu[t, :, :] =  flux_qdir+ flux_qsur + flux_qint+flux_qbfp+flux_qbfs

            ETmu[t, :, :] = flux_euztw+flux_euzfw+flux_elztw



        ETave = ETmu.mean(-1, keepdim=True)

        # get the initial average
        if muwts is None: # simple average
            Qsimave = Qsimmu.mean(-1)
        else: # weighted average using learned weights
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            # scale two routing parameters
            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations
            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True: # output states
            return Qs, S1,S2,S3,S4,S5
        else:
            # return Qs
            Qall = torch.cat((Qs, ETave), dim=-1)
            return Qall

    def split_1(self, p1, In):
        """
        Split flow (returns flux [mm/d])

        :param p1: fraction of flux to be diverted [-]
        :param In: incoming flux [mm/d]
        :return: divided flux
        """
        out = p1 * In
        return out

    def soilmoisture_1(self, S1, S1max, S2, S2max):
        """
        Water rebalance to equal relative storage (2 stores)

        :param S1: current storage in S1 [mm]
        :param S1max: maximum storage in S1 [mm]
        :param S2: current storage in S2 [mm]
        :param S2max: maximum storage in S2 [mm]
        :return: rebalanced water storage
        """
        mask = S1/S1max < S2/S2max
        mask = mask.type(torch.cuda.FloatTensor)
        out = ((S2 * S1max - S1 * S2max) / (S1max + S2max)) * mask
        return out

    def evap_7(self, S, Smax, Ep, dt):
        """
        Evaporation scaled by relative storage

        :param S: current storage [mm]
        :param Smax: maximum contributing storage [mm]
        :param Ep: potential evapotranspiration rate [mm/d]
        :param dt: time step size [d]
        :return: evaporation [mm]
        """
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def saturation_1(self, In, S, Smax):
        """
        Saturation excess from a store that has reached maximum capacity

        :param In: incoming flux [mm/d]
        :param S: current storage [mm]
        :param Smax: maximum storage [mm]
        :param args: smoothing variables (optional)
        :return: saturation excess
        """
        mask = S >= Smax
        mask = mask.type(torch.cuda.FloatTensor)
        out = In * mask

        return out

    def interflow_5(self, p1, S):
        """
        Linear interflow

        :param p1: time coefficient [d-1]
        :param S: current storage [mm]
        :return: interflow output
        """
        out = p1 * S
        return out

    def evap_1(self, S, Ep, dt):
        """
        Evaporation at the potential rate

        :param S: current storage [mm]
        :param Ep: potential evaporation rate [mm/d]
        :param dt: time step size
        :return: evaporation output
        """
        out = torch.min(S / dt, Ep)
        return out

    def percolation_4(self, p1, p2, p3, p4, p5, S, Smax, dt):
        """
        Demand-based percolation scaled by available moisture

        :param p1: base percolation rate [mm/d]
        :param p2: percolation rate increase due to moisture deficiencies [mm/d]
        :param p3: non-linearity parameter [-]
        :param p4: summed deficiency across all model stores [mm]
        :param p5: summed capacity of model stores [mm]
        :param S: current storage in the supplying store [mm]
        :param Smax: maximum storage in the supplying store [mm]
        :param dt: time step size [d]
        :return: percolation output
        """
        # Prevent negative S values and ensure non-negative percolation demands
        S_rel = torch.max(torch.tensor(1e-8).cuda(), S / Smax)

        percolation_demand = p1 * (torch.tensor(1.0).cuda() + p2 * (p4 / p5) ** (torch.tensor(1.0).cuda() + p3))
        out = torch.max(torch.tensor(1e-8).cuda(), torch.min(S / dt, S_rel * percolation_demand))
        return out

    def soilmoisture_2(self, S1, S1max, S2, S2max, S3, S3max):
        """
        Water rebalance to equal relative storage (3 stores)

        :param S1: current storage in S1 [mm]
        :param S1max: maximum storage in S1 [mm]
        :param S2: current storage in S2 [mm]
        :param S2max: maximum storage in S2 [mm]
        :param S3: current storage in S3 [mm]
        :param S3max: maximum storage in S3 [mm]
        :return: rebalanced water storage
        """
        part1 = S2 * (S1 * (S2max + S3max) + S1max * (S2 + S3)) / ((S2max + S3max) * (S1max + S2max + S3max))
        mask = S1 / S1max < (S2 + S3) / (S2max + S3max)
        mask = mask.type(torch.cuda.FloatTensor)
        out = part1 * mask
        return out

    def baseflow_1(self,K,S):
        return K * S



    def deficitBasedDistribution_pytorch(self, S1, S1max, S2, S2max):
        # Calculate relative deficits
        rd1 = (S1max-S1 ) / S1max
        rd2 = (S2max-S2 ) / S2max

        # Calculate fractional split
        total_rd = rd1 + rd2
        mask = total_rd != torch.tensor(0.0).cuda()
        mask = mask.type(torch.cuda.FloatTensor)
        total_max = S1max + S2max
        f1 = rd1 / total_rd * mask + S1max / total_max*(torch.tensor(1.0)-mask)

        return f1


def UH_conv(x,UH,viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1],m[2]])
    aa = F.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0) # scale to 1 for each UH

    return w