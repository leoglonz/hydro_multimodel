import torch
from models.pet_models.potet import get_potet
import torch.nn.functional as F


class HBVMulTDET(torch.nn.Module):
    """
    HBV Model Pytorch version (dynamic and static param capable) adapted from
    dPL_Hydro_SNTEMP @ Farshid Rahmani.

    Modified from the original numpy version from Beck et al., 2020
    (http://www.gloh2o.org/hbv/), which runs the HBV-light hydrological model
    (Seibert, 2005).
    """
    def __init__(self, config):
        super(HBVMulTDET, self).__init__()
        self.parameters_bound = dict(parBETA=[1.0, 6.0],
                                     parFC=[50, 1000],
                                     parK0=[0.05, 0.9],
                                     parK1=[0.01, 0.5],
                                     parK2=[0.001, 0.2],
                                     parLP=[0.2, 1],
                                     parPERC=[0, 10],
                                     parUZL=[0, 100],
                                     parTT=[-2.5, 2.5],
                                     parCFMAX=[0.5, 10],
                                     parCFR=[0, 0.1],
                                     parCWH=[0, 0.2],
                                     parBETAET=[0.3, 5],
                                     parC=[0, 1]
                                     )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]

    def UH_gamma(self, a, b, lenF=10):
        # UH. a [time (same all time steps), batch, var]
        m = a.shape
        lenF = min(a.shape[0], lenF)
        w = torch.zeros([lenF, m[1], m[2]])
        aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
        theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
        t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
        t = t.cuda(aa.device)
        denom = (aa.lgamma().exp()) * (theta ** aa)
        mid = t ** (aa - 1)
        right = torch.exp(-t / theta)
        w = 1 / denom * mid * right
        w = w / w.sum(0)  # scale to 1 for each UH

        return w

    def UH_conv(self, x, UH, viewmode=1):
        """
        UH is a vector indicating the unit hydrograph
        the convolved dimension will be the last dimension
        UH convolution is
        Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
        conv1d does \integral(w(\tao)*x(t+\tao))d\tao
        hence we flip the UH
        https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
        view
        x: [batch, var, time]
        UH:[batch, var, uhLen]
        batch needs to be accommodated by channels and we make use of groups
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        https://pytorch.org/docs/stable/nn.functional.html
        """
        mm = x.shape;
        nb = mm[0]
        m = UH.shape[-1]
        padd = m - 1
        if viewmode == 1:
            xx = x.view([1, nb, mm[-1]])
            w = UH.view([nb, 1, m])
            groups = nb

        y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
        if padd != 0:
            y = y[:, :, 0:-padd]
        return y.view(mm)

    def source_flow_calculation(self, args, flow_out, c_NN, after_routing=True):
        varC_NN = args['var_c_nn']
        if 'DRAIN_SQKM' in varC_NN:
            area_name = 'DRAIN_SQKM'
        elif 'area_gages2' in varC_NN:
            area_name = 'area_gages2'
        else:
            print("area of basins are not available among attributes dataset")
        area = c_NN[:, varC_NN.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
            flow_out['flow_sim'].shape[
                0], 1, 1)
        # flow calculation. converting mm/day to m3/sec
        if after_routing == True:
            srflow = (1000 / 86400) * area * (flow_out['srflow']).repeat(1, 1, args['nmul'])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out['ssflow']).repeat(1, 1, args['nmul'])  # ras
            gwflow = (1000 / 86400) * area * (flow_out['gwflow']).repeat(1, 1, args['nmul'])
        else:
            srflow = (1000 / 86400) * area * (flow_out['srflow_no_rout']).repeat(1, 1, args['nmul'])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out['ssflow_no_rout']).repeat(1, 1, args['nmul'])  # ras
            gwflow = (1000 / 86400) * area * (flow_out['gwflow_no_rout']).repeat(1, 1, args['nmul'])
        # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
        # ssflow = torch.clamp(ssflow, min=0.0)
        # gwflow = torch.clamp(gwflow, min=0.0)
        return srflow, ssflow, gwflow

    def param_bounds_2D(self, params, num, bounds, ndays, nmul):
        out_temp = (
                params[:, num * nmul: (num + 1) * nmul]
                * (bounds[1] - bounds[0])
                + bounds[0]
        )
        out = out_temp.unsqueeze(0).repeat(ndays, 1, 1).reshape(
            ndays, params.shape[0], nmul
        )
        return out

    def change_param_range(self, param, bounds):
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

    def forward(self, x_hydro_model, c_hydro_model, params_raw, args, muwts=None, warm_up=0,init=False, routing=False, comprout=False, conv_params_hydro=None):
        nearzero = args['nearzero']
        nmul = args['nmul']

        # Initialization
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                initmodel = HBVMulTDET(args).to(args['device'])
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, c_hydro_model, params_raw, args,
                                                                      muwts=None, warm_up=0, init=True, routing=False,
                                                                      comprout=False, conv_params_hydro=None)
        else:
            # Without buff time, initialize state variables with zeros
            Ngrid = x_hydro_model.shape[1]
            SNOWPACK = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            MELTWATER = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            SM = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            SUZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            SLZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        # Parameters
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = self.change_param_range(param=params_raw[:, :, num, :],
                                                             bounds=self.parameters_bound[param])

        vars = args['observations']['var_t_hydro_model']
        vars_c = args['observations']['var_c_hydro_model']
        P = x_hydro_model[warm_up:, :, vars.index('prcp(mm/day)')]
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        mean_air_temp = x_hydro_model[warm_up:, :, vars.index('tmean(C)')].unsqueeze(2).repeat(1, 1, nmul)

        if args['pet_module'] == 'potet_hamon':
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.004, 0.008], ndays=No_days, nmul=args['nmul'])
            PET = get_potet(
                args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=PET_coef
            )  # mm/day
        elif args['pet_module'] == 'potet_hargreaves':
            day_of_year = x_hydro_model[warm_up:, :, vars.index('dayofyear')].unsqueeze(-1).repeat(1, 1, nmul)
            lat = c_hydro_model[:, vars_c.index('lat')].unsqueeze(0).unsqueeze(-1).repeat(day_of_year.shape[0], 1, nmul)
            Tmaxf = x_hydro_model[warm_up:, :, vars.index('tmax(C)')].unsqueeze(2).repeat(1, 1, nmul)
            Tminf = x_hydro_model[warm_up:, :, vars.index('tmin(C)')].unsqueeze(2).repeat(1, 1, nmul)
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
            #                                   nmul=args['nmul'])

            PET = get_potet(
                args=args, tmin=Tminf, tmax=Tmaxf,
                tmean=mean_air_temp, lat=lat,
                day_of_year=day_of_year
            )
            # AET = PET_coef * PET     # here PET_coef converts PET to Actual ET here
        elif args['pet_module'] == 'dataset':
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
            #                                 nmul=args['nmul'])
            # here PET_coef converts PET to Actual ET
            PET = x_hydro_model[warm_up:, :, vars.index(args['pet_dataset_name'])].unsqueeze(-1).repeat(1, 1, nmul)
            # AET = PET_coef * PET

        Nstep, Ngrid = P.size()


        # # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        # parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static para matrix
        # parhbvFull = torch.clone(parstaFull)
        # # create probability mask for each parameter on the basin dimension
        # pmat = torch.ones([1, Ngrid, 1])*dydrop
        # for ix in tdlst:
        #     staPar = parstaFull[:, :, ix-1, :]
        #     dynPar = parAllTrans[:, :, ix-1, :]
        #     drmask = torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
        #     comPar = dynPar*(1-drmask) + staPar*drmask
        #     parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(args['device'])
        Q0_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args['device'])
        Q1_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args['device'])
        Q2_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args['device'])

        AET = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        recharge_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        excs_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        evapfactor_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        tosoil_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        PERC_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        SWE_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])
        capillary_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(args['device'])

        # Do static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in args['dy_params']['HBV']:  ## it is a static parameter
                params_dict[key] = params_dict_raw[key][-1, :, :]

        # Do dynamic parameters based on dydrop ratio.
        # (Drops dynamic params for some basins (based on ratio), and substitutes
        # them for a static params, which is set to the value of the param on the
        # last day of data.
        if len(args['dy_params']['HBV_capillary']) > 0:
            params_dict_raw_dy = dict()
            pmat = torch.ones([Ngrid, 1]) * args['dy_drop']
            for i, key in enumerate(args['dy_params']['HBV_capillary']):
                drmask = torch.bernoulli(pmat).detach_().to(args['device'])
                dynPar = params_dict_raw[key]
                staPar = params_dict_raw[key][-1, :, :].unsqueeze(0).repeat([dynPar.shape[0], 1, 1])
                params_dict_raw_dy[key] = dynPar * (1 - drmask) + staPar * drmask

        for t in range(Nstep):
            # Treat dynamic parameters
            for key in params_dict_raw.keys():
                if key in args['dy_params']['HBV_capillary']:  ## it is a dynamic parameter
                    # params_dict[key] = params_dict_raw[key][warm_up + t, :, :]
                      # to drop dynamic parameters as static in some basins
                    params_dict[key] = params_dict_raw_dy[key][warm_up + t, :, :]

                # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (mean_air_temp[t, :, :] >= params_dict['parTT']).type(torch.float32))
            SNOW = torch.mul(PRECIP, (mean_air_temp[t, :, :] < params_dict['parTT']).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = params_dict['parCFMAX'] * (mean_air_temp[t, :, :] - params_dict['parTT'])
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = torch.clamp(SNOWPACK - melt, min=nearzero)
            refreezing = params_dict['parCFR'] * params_dict['parCFMAX'] * (
                params_dict['parTT'] - mean_air_temp[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = torch.clamp(MELTWATER - refreezing, min=nearzero)
            tosoil = MELTWATER - (params_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = torch.clamp(MELTWATER - tosoil, min=nearzero)

            # Soil and evaporation
            soil_wetness = (SM / params_dict['parFC']) ** params_dict['parBETA']
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge
            excess = SM - params_dict['parFC']
            excess = torch.clamp(excess, min=0.0)
            SM = torch.clamp(SM - excess, min=nearzero)

            # NOTE: Different from HBVmul. Add an ET shape parameter parBETAET. this param can be static or dynamic
            evapfactor = (SM / (params_dict['parLP'] * params_dict['parFC'])) ** params_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PET[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            AET[t, :, :] = ETact
            SM = torch.clamp(SM - ETact, min=nearzero)  # SM can not be zero for gradient tracking.
            capillary = torch.min(SLZ, params_dict['parC'] * SLZ * (1.0 - torch.clamp(SM / params_dict['parFC'], max=1.0)))

            SM = torch.clamp(SM + capillary, min=nearzero)
            SLZ = torch.clamp(SLZ - capillary, min=nearzero)

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, params_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = params_dict['parK0'] * torch.clamp(SUZ - params_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = params_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = params_dict['parK2'] * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2

            recharge_sim[t, :, :] = recharge
            excs_sim[t, :, :] = excess
            evapfactor_sim[t, :, :] = evapfactor
            tosoil_sim[t, :, :] = tosoil
            PERC_sim[t, :, :] = PERC
            SWE_sim[t, :, :] = SNOWPACK
            capillary_sim[t, :, :] = capillary

        # get the primary average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routing is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * nmul)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = self.change_param_range(param=conv_params_hydro[:, 0],
                                            bounds=self.conv_routing_hydro_model_bound[0])
            tempb = self.change_param_range(param=conv_params_hydro[:, 1],
                                            bounds=self.conv_routing_hydro_model_bound[1])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = self.UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = self.UH_conv(rf, UH).permute([2, 0, 1])
            # do routing individually for Q0, Q1, and Q2
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            Q0_rout = self.UH_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            Q1_rout = self.UH_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            Q2_rout = self.UH_conv(rf_Q2, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, nmul)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:  # no routing, output the primary average simulations
            Qs = torch.unsqueeze(Qsimave, -1)  # add a dimension

        if init is True:  # means we are in warm up
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return dict(flow_sim=Qsrout,
                        srflow=Q0_rout,
                        ssflow=Q1_rout,
                        gwflow=Q2_rout,
                        AET_hydro=AET.mean(-1, keepdim=True),
                        PET_hydro=PET.mean(-1, keepdim=True),
                        flow_sim_no_rout=Qsim.mean(-1, keepdim=True),
                        srflow_no_rout=Q0_sim.mean(-1, keepdim=True),
                        ssflow_no_rout=Q1_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=Q2_sim.mean(-1, keepdim=True),
                        recharge=recharge_sim.mean(-1, keepdim=True),
                        excs=excs_sim.mean(-1, keepdim=True),
                        evapfactor=evapfactor_sim.mean(-1, keepdim=True),
                        tosoil=tosoil_sim.mean(-1, keepdim=True),
                        percolation=PERC_sim.mean(-1, keepdim=True),
                        SWE=SWE_sim.mean(-1, keepdim=True),
                        capillary=capillary_sim.mean(-1, keepdim=True)
                        )
