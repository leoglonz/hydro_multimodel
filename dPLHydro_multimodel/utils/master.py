"""
General scripts for running multimodel interface.

May decide to organize these later, but for now this file includes
everything not in functional.py, and not in existing files.
"""
import json
import os
import platform

import numpy as np
import torch
from data.load_data.time import tRange2Array

# Set list of supported hydro models here:
supported_models = ['HBV', 'dPLHBV_stat', 'dPLHBV_dyn', 'SACSMA', 'SACSMA_snow',
                   'marrmot_PRMS']


def set_globals():
    """
    Select torch device and dtype global vars per user system.
    """ 
    global device, dtype

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available():
        # Use Mac M-series ARM architecture.
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    return device, dtype



def set_platform_dir():
    """
    Set output directory path to for systems with directory structures
    and locations.
    Currently supports: windows, mac os, and linux colab.

    outputs
        dir: output directory
    """
    if platform.system() == 'Windows':
        # Windows
        dir = os.path.join('D:\\','code_repos','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Darwin':
        # MacOs
        dir = os.path.join('Users','leoglonz','Desktop','water','data','model_runs','hydro_multimodel_results')
    elif platform.system() == 'Linux':
        # For Colab
        dir = os.path.join('content','drive','MyDrive','Colab','data','model_runs','hydro_multimodel_results')
    else:
        raise ValueError('Unsupported operating system.')
    
    return dir



def get_model_dict(modList):
    """
    Create model and argument dictionaries to individual manage models in an
    ensemble interface.
    
    Inputs:
        modList: list of models.
    """
    models, arg_list = {}, {}
    for mod in modList:
        if mod in supported_models:
            models[mod] = None
            arg_list[mod] = config[mod]
        else:
            raise ValueError(f"Unsupported model type", mod)
    return models, arg_list



def create_tensor(dims, requires_grad=False):
    """
    A small function to centrally manage device, data types, etc., of new arrays.
    """
    return torch.zeros(dims,requires_grad=requires_grad,dtype=dtype).to(device)



def create_dict_from_keys(keyList, mtd=0, dims=None, dat=None):
    """
    A modular dictionary initializer from C. Shen.

    mtd = 
        0: Init keys to None,
        1: Init keys to zero tensors,
        11: Init keys to tensors with the same vals as `dat`,
        2: Init keys to slices of `dat`,
        21: Init keys with cloned slices of `dat`.
    """
    d = {}
    for kk, k in enumerate(keyList):
        if mtd == 0 or mtd is None or mtd == 'None':
            d[k] = None
        elif mtd == 1 or mtd == 'zeros':
            d[k] = create_tensor(dims)
        elif mtd == 11 or mtd == 'value':
            d[k] = create_tensor(dims) + dat
        elif mtd == 2 or mtd == 'ref':
            d[k] = dat[..., kk]
        elif mtd == 21 or mtd == 'refClone':
            d[k] = dat[..., kk].clone()
    return d


# def save_output(args_list, preds, y_obs, out_dir):
#     """
#     Save extracted test preds and obs for all models.
#     """
#     for i, mod in enumerate(args_list):
#         if i == 0:
#             multim_dir = str(mod)
#         else:
#             multim_dir += '_' + str(mod)

#         out_dir = os.path.join(out_dir, multim_dir)
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir, exist_ok=True)

#         arg = args_list[list(args_list)[1]]
#         dir = 'multim_E' + str(arg['epochs']) + '_B' + str(arg['batch_size']) + '_R' + str(arg['rho']) +  '_BT' + str(arg['warm_up']) + '_H' + str(arg['hidden_size']) + '_tr1980_1995_n' + str(arg['nmul'])

#         np.save(os.path.join(out_dir, 'preds_' + dir + '.npy'), preds)
#         np.save(os.path.join(out_dir, 'obs_' + dir + '.npy'), y_obs)


def save_outputs(args, list_out_diff_model, y_obs, calculate_metrics=True):
    for key in list_out_diff_model[0].keys():
        if len(list_out_diff_model[0][key].shape) == 3:
            dim = 1
        else:
            dim = 0
        concatenated_tensor = torch.cat([d[key] for d in list_out_diff_model], dim=dim)
        file_name = key + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in args["target"]:
        item_obs = y_obs[:, :, args["target"].index(var)]
        file_name = var + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), item_obs)

    if calculate_metrics == True:
        predLst = list()
        obsLst = list()
        name_list = []
        if args["hydro_model_name"] != "None":
            flow_sim = torch.cat([d["flow_sim"] for d in list_out_diff_model], dim=1)
            flow_obs = y_obs[:, :, args["target"].index("00060_Mean")]
            predLst.append(flow_sim.numpy())
            obsLst.append(np.expand_dims(flow_obs, 2))
            name_list.append("flow")
        if args["temp_model_name"] != "None":
            temp_sim = torch.cat([d["temp_sim"] for d in list_out_diff_model], dim=1)
            predLst.append(temp_sim.numpy())
            if "00010_Mean" in args["target"]:  # this line helps have flow_temp model with only flow in loss func
                temp_obs = y_obs[:, :, args["target"].index("00010_Mean")]
                obsLst.append(np.expand_dims(temp_obs, 2))
                name_list.append("temp")

        # we need to swap axes here to have [basin, days]
        statDictLst = [
            stat.statError(np.swapaxes(x.squeeze(), 1, 0), np.swapaxes(y.squeeze(), 1, 0))
            for (x, y) in zip(predLst, obsLst)
        ]
        ### save this file
        # median and STD calculation
        for st, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(st), 3])
            for key in st.keys():
                median = np.nanmedian(st[key])  # abs(i)
                STD = np.nanstd(st[key])  # abs(i)
                mean = np.nanmean(st[key])  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(
                mdstd, index=st.keys(), columns=["median", "STD", "mean"]
            )
            mdstd.to_csv((os.path.join(args["out_dir"], args["testing_dir"], "mdstd_" + name + ".csv")))

            # Show boxplots of the results
            # plt.rcParams["font.size"] = 14
            # keyLst = ["Bias", "RMSE", "ubRMSE", "NSE", "Corr"]
            # dataBox = list()
            # for iS in range(len(keyLst)):
            #     statStr = keyLst[iS]
            #     temp = list()
            #     # for k in range(len(st)):
            #     data = st[statStr]
            #     data = data[~np.isnan(data)]
            #     temp.append(data)
            #     dataBox.append(temp)
            # labelname = [
            #     "Hybrid differentiable model"
            # ]  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

            # xlabel = ["Bias ($\mathregular{deg}$C)", "RMSE", "ubRMSE", "NSE", "Corr"]
            # fig = plot.plotBoxFig(
            #     dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8)
            # )
            # fig.patch.set_facecolor("white")
            # boxPlotName = "PGML"
            # fig.suptitle(boxPlotName, fontsize=12)
            # plt.rcParams["font.size"] = 12
            # plt.savefig(
            #     os.path.join(args["out_dir"], args["testing_dir"], "Box_" + name + ".png")
            # )  # , dpi=500
            # # fig.show()
            # plt.close()


def create_output_dirs(args):
    # Checking the directory
    os.makedirs(args['output_dir'], exist_ok=True)

    out_folder = args['nn_model'] + \
             '_E' + str(args['epochs']) + \
             '_R' + str(args['rho']) + \
             '_B' + str(args['batch_size']) + \
             '_H' + str(args['hidden_size']) + \
             '_n' + str(args['nmul']) + \
             '_' + str(args['random_seed'])

    os.makedirs(os.path.join(args['output_dir'], out_folder), exist_ok=True)

    ## make a folder for static and dynamic parametrization
    if args['dyn_hydro_params']['HBV'] != []:
        dyn_params = 'dynamic_para'
    else:
        dyn_params = 'static_para'

    testing_dir = 'test_results'
    
    os.makedirs(os.path.join(args['output_dir'], out_folder, dyn_params, testing_dir), 
                exist_ok=True)
    
    args['output_dir'] = os.path.join(args['output_dir'], out_folder, dyn_params)
    args['testing_dir'] = testing_dir

    # saving the args file in output directory
    config_file = json.dumps(args)
    config_path = os.path.join(args['output_dir'], 'config_file.json')
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, 'w')
    f.write(config_file)
    f.close()

    return args



# def loadModel(out, epoch=None):
#     if epoch is None:
#         mDict = readMasterFile(out)
#         epoch = mDict['train']['nEpoch']
#     model = hydroDL.model.train.loadModel(out, epoch)
#     return model


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model
