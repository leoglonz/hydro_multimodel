import torch
from hydroDL.model.rnn import UH_gamma,UH_conv
import numpy as np


#***************************************#
# Regional flow =sum_j ( tanh((Ac-Ac*)/2000) *Trbound)
# Parameter range Trbound [0,20]; Ac* [0,50000]
# Regional flow parameters are from additional NN and caculated at the merit level
# Change threshold formula to the linear function
#***************************************#


class HBVMulET_water_loss(torch.nn.Module):
    """Multi-component HBV Model PyTorch version"""
    # Add an ET shape parameter; others are the same as class HBVMul()
    # refer HBVMul() for detailed comments

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulET_water_loss, self).__init__()




    def forward(self, x, parameters,waterloss_parameters,Ai_batch , Ac_batch , idx_matric , mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False):

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            raise Exception("This function currently does not allow bufftime>0")
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)


        ## scale the parameters to real values from [0,1]

        parascaLst1 = [[1,6], [0.05,0.9],[0.3,5]]  # HBV para
            
        routscaLst = [[0,2.9], [0,6.5]]

        parascaLst2 = [[50,1000], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2],[0,1], [0,20], [0, 2500]]

        paraLst1 = []
        for ip in range(len(parascaLst1)): # not include routing. Scaling the parameters
            paraLst1.append( parascaLst1[ip][0] + parameters[:,ip,:]*(parascaLst1[ip][1]-parascaLst1[ip][0]) )
        paraLst2 = []
        for ip in range(len(parascaLst2)): # not include routing. Scaling the parameters
            paraLst2.append( parascaLst2[ip][0] + waterloss_parameters[:,ip,:]*(parascaLst2[ip][1]-parascaLst2[ip][0]) )

        parBETA,parK0, parBETAET = paraLst1
        parFC,  parK1, parK2, parLP, parPERC,parUZL,  parTT, parCFMAX, parCFR, parCWH,parC,parTRbound,parAc = paraLst2

        Nstep, Ngrid = P.size()

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(x)
        
        mu2 = parAc.shape[-1]
        Ac_batch_torch = (torch.from_numpy(np.array(Ac_batch)).to(x)).unsqueeze(-1).repeat(1,mu2)
        # Ai_batch_torch = (torch.from_numpy(np.array(Ai_batch)).to(x)).unsqueeze(-1).repeat(1,mu2)
        # idx_matric_expand =  (torch.from_numpy(idx_matric).to(x)).unsqueeze(1).repeat(1,mu2)
        
        
        
       # parAscale_expand = parAscale.unsqueeze(0).repeat(len(Ai_batch),1,1)
        
        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness


            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY HERE. Add the shape para parBETAET for ET equation
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            # evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            capillary = torch.min(SLZ, parC * SLZ * (1.0 - torch.clamp(SM / parFC, max=1.0)))

            SM = torch.clamp(SM + capillary, min=PRECS)
            SLZ = torch.clamp(SLZ - capillary, min=PRECS)

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC

            regional_flow = torch.clamp((Ac_batch_torch-parAc)/1000,min = -1, max = 1) * parTRbound*(Ac_batch_torch<2500)+\
            torch.exp(torch.clamp(-(Ac_batch_torch-2500)/50, min = -10.0,max = 0.0))* parTRbound*(Ac_batch_torch>=2500)
            #regional_flow = (RT.unsqueeze(-1).repeat(1,1,Ngrid) *idx_matric_expand).sum(0)
            #regional_flow = torch.clamp(regional_flow0, max = 0.0).unsqueeze(-1).repeat(1,mu)
            
            SLZ = torch.clamp(SLZ + regional_flow, min=0.0)
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2


            Qsimmu[t, :, :] = Q0 + Q1 + Q2


        # get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

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

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs # total streamflow

class HBVMulTDET_water_loss(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_water_loss, self).__init__()





    def forward(self, x, parameters,waterloss_parameters, Ai_batch , Ac_batch , idx_matric, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
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
        #bufftime = 0
        # Initialization to warm-up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET_water_loss()
                buffpara = parameters[bufftime-1, :, :, :]
                
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, buffpara,waterloss_parameters,Ai_batch , Ac_batch , idx_matric , mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu) # precip
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET
        parAll = parameters[bufftime:, :, :, :]
        
        
        ########### NOTE: Check whether waterloss_parameters had its warmup removed
        parhbvFull = torch.zeros_like(parAll)
        parWaterLoss = torch.zeros_like(waterloss_parameters)
        ## scale the parameters to real values from [0,1]

        hbvscaLst1 = [[1,6], [0.05,0.9],  [0.3,5]]  # HBV para

        routscaLst = [[0,2.9], [0,6.5]]  # routing para
        
        hbvscaLst2 = [ [50,1000], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2],[0,1] ,[0,20],[0,2500]]
        
        for ip in range(len(hbvscaLst1)): # not include routing. Scaling the parameters
            parhbvFull[:,:,ip,:] = hbvscaLst1[ip][0] + parAll[:,:,ip,:]*(hbvscaLst1[ip][1]-hbvscaLst1[ip][0])

        for ip in range(len(hbvscaLst2)): # not include routing. Scaling the parameters
            parWaterLoss[:,ip] = hbvscaLst2[ip][0] + waterloss_parameters[:,ip]*(hbvscaLst2[ip][1]-hbvscaLst2[ip][0])


        Nstep, _ = P.size()
        Ngrid = idx_matric.shape[-1]
        # Initialize time series of model variables to save results
        Qsimmu = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(parhbvFull)
        ETmu = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(parhbvFull)
        SWEmu = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0 = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(parhbvFull)
        Qsimmu1 = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(parhbvFull)
        Qsimmu2 = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())



        
        paraLst2 = []
        for ip in range(len(hbvscaLst2)):  # unpack water loss parameters
            paraLst2.append(parWaterLoss[:, ip,:])
            
        parFC, parK1, parK2, parLP, parPERC, parUZL,parTT, parCFMAX, parCFR, parCWH, parC, parTRbound,parAc = paraLst2

        mu2 = parAc.shape[-1]
        Ac_batch_torch = (torch.from_numpy(np.array(Ac_batch)).to(x)).unsqueeze(-1).repeat(1,mu2)
        # Ai_batch_torch = (torch.from_numpy(np.array(Ai_batch)).to(x)).unsqueeze(-1).repeat(1,mu2)
        # Ai_batch_expand = (torch.from_numpy(np.array(Ai_batch)).to(x)).unsqueeze(0).repeat(Nstep,1)
        # idx_matric_expand = ( torch.from_numpy(idx_matric).to(x)).unsqueeze(0).repeat(Nstep,1,1)  
        Ai_batch_torch = torch.from_numpy(np.array(Ai_batch)).to(x)
        idx_matric_torch = torch.from_numpy(np.array(idx_matric)).to(x)
        


        for t in range(Nstep):
            paraLst1 = []
            for ip in range(len(hbvscaLst1)):  # unpack HBV parameters
                paraLst1.append(parhbvFull[t, :, ip, :])



            parBETA,  parK0,  parBETAET = paraLst1
         
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            TEMP = Tm[t, :, :]
            EVAP_P = ETpm[t, :, :]

            RAIN = torch.mul(PRECIP, (TEMP >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (TEMP < parTT).type(torch.float32))

            # Snow process
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (TEMP - parTT)
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - TEMP)
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = EVAP_P  * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            capillary = torch.min(SLZ, parC * SLZ * (1.0 - torch.clamp(SM / parFC, max=1.0)))

            SM = torch.clamp(SM + capillary, min=PRECS)
            SLZ = torch.clamp(SLZ - capillary, min=PRECS)

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            
            regional_flow = torch.clamp((Ac_batch_torch-parAc)/1000,min = -1, max = 1) * parTRbound*(Ac_batch_torch<2500)+\
            torch.exp(torch.clamp(-(Ac_batch_torch-2500)/50, min = -10.0,max = 0.0))* parTRbound*(Ac_batch_torch>=2500)
            # RT = (torch.clamp((Ac_batch_torch-parAc)/2000 * 0.5 ,min = -1, max = 1) * parTRbound*Ai_batch_torch).mean(-1)
            # regional_flow = (RT.unsqueeze(-1).repeat(1,Ngrid) *idx_matric_expand).sum(0).unsqueeze(-1).repeat(1,mu)
            #regional_flow = torch.clamp(regional_flow0, max = 0.0).unsqueeze(-1).repeat(1,mu)


            SLZ = torch.clamp(SLZ + regional_flow, min=0.0)
            regional_flow_out =  torch.max(regional_flow, -SLZ)

            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2




            Qsimmu[t, :] = (((Q0 + Q1 + Q2).mean(-1) * Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0)
                
            # save components for Q

            Qsimmu0[t, :] = (((regional_flow_out).mean(-1) * Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0)
            Qsimmu1[t, :] = (((Q1).mean(-1) * Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0)
            Qsimmu2[t, :] = (((Q2).mean(-1) * Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0)
            ETmu[t, :] = (((ETact).mean(-1) * Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0) 
            SWEmu[t, :] = (((SNOWPACK).mean(-1) * Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0)


        if routOpt is True: # routing


            # scale two routing parameters
            tempa0 = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb0 = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            tempa =  ((tempa0* Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0) 
            tempb =  ((tempb0* Ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matric_torch).sum(0) 

            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsimmu, -1).permute([1, 2, 0])   # dim:gage*var*time
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
            Qs = torch.unsqueeze(Qsimmu, -1) # add a dimension




        if outstate is True: # output states
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # return Qs
            Qall = torch.cat((Qs, Qsimmu0.unsqueeze(-1), Qsimmu1.unsqueeze(-1), Qsimmu2.unsqueeze(-1), ETmu.unsqueeze(-1), SWEmu.unsqueeze(-1), Qsimmu.unsqueeze(-1)), dim=-1)
            return Qall
