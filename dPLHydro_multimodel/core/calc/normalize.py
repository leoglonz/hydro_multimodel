import os
import numpy as np
import json
import xarray as xr
import torch
from tqdm import tqdm
from typing import Dict



def calc_stat_basinnorm(y, c, config) -> list:  
    """
    Taken from the calStatsbasinnorm function of hydroDL.
    For daily streamflow normalized by basin area and precipitation.

    :param y: streamflow data to be normalized
    :param x: x is forcing+attr numpy matrix
    :param config: config file
    :return: statistics to be used for flow normalization
    """
    y[y == (-999)] = np.nan
    y[y < 0] = 0
    attr_list = config['observations']['var_c_nn']
    # attr_data = read_attr_data(config, idLst=idLst)
    if 'DRAIN_SQKM' in attr_list:
        area_name = 'DRAIN_SQKM'
    elif 'area_gages2' in attr_list:
        area_name = 'area_gages2'
    basinarea = c[:, attr_list.index(area_name)]  #  'DRAIN_SQKM'
    if 'PPTAVG_BASIN' in attr_list:
        p_mean_name = 'PPTAVG_BASIN'
    elif 'p_mean' in attr_list:
        p_mean_name = 'p_mean'
    meanprep = c[:, attr_list.index(p_mean_name)]  #   'PPTAVG_BASIN'
    temparea = np.repeat(np.expand_dims(basinarea, axis=(1,2)), y.shape[0]).reshape(y.shape)
    tempprep = np.repeat(np.expand_dims(meanprep, axis=(1, 2)), y.shape[0]).reshape(y.shape)
    flowua = (y * 0.0283168 * 3600 * 24) / (
        (temparea * (10**6)) * (tempprep * 10 ** (-2)) / 365
    )  # unit (m^3/day)/(m^3/day)
    a = flowua.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics plus 0.1 for 0 values
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calculate_statistics(x) -> list:
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


# TODO: Eventually replace calculate_statistics with the version below.
def calculate_statistics_dmc(data: xr.Dataset, column: str = "time", row: str = "gage_id") -> Dict[str, torch.Tensor]:
    """
    calculating statistics for the data in a similar manner to calStat from hydroDL
    """
    statistics = {}
    p_10 = data.quantile(0.1, dim=column)
    p_90 = data.quantile(0.9, dim=column)
    mean = data.mean(dim=column)
    std = data.std(dim=column)
    col_names = data[row].values.tolist()
    for idx, col in enumerate(
        tqdm(col_names, desc="\rCalculating statistics", ncols=140, ascii=True)
    ):
        col_str = str(col)
        statistics[col_str] = torch.tensor(
            data=[
                p_10.streamflow.values[idx],
                p_90.streamflow.values[idx],
                mean.streamflow.values[idx],
                std.streamflow.values[idx],
            ]
        )
    return statistics


def calculate_statistics_gamma(x) -> list:
    """
    Taken from the calStatgamma function of hydroDL.
    For daily streamflow and precipitation.
    """
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calculate_statistics_all(config, x, c, y) -> None:
    """
    Taken from the calStatAll function of hydroDL.
    """
    statDict = dict()
    # target
    for i, target_name in enumerate(config['target']):
        # calculating especialized statistics for streamflow
        if target_name == '00060_Mean':
            statDict[config['target'][i]] = calc_stat_basinnorm(y[:, :, i: i+1], c, config)
        else:
            statDict[config['target'][i]] = calculate_statistics(y[:, :, i: i+1])

    # forcing
    varList = config['observations']['var_t_nn']
    for k in range(len(varList)):
        var = varList[k]
        if var == 'prcp(mm/day)':
            statDict[var] = calculate_statistics_gamma(x[:, :, k])
        elif (var == '00060_Mean') or (var == 'combine_discharge'):
            statDict[var] = calc_stat_basinnorm(x[:, :, k: k + 1], x, config)
        else:
            statDict[var] = calculate_statistics(x[:, :, k])
    # attributes
    varList = config['observations']['var_c_nn']
    for k, var in enumerate(varList):
        statDict[var] = calculate_statistics(c[:, k])

    statFile = os.path.join(config['output_dir'], 'Statistics_basinnorm.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)


def trans_norm(config, x, varLst, *, toNorm) -> np.array:
    """
    Taken from the transNorm function of hydroDL.
    """
    statFile = os.path.join(config['output_dir'], 'Statistics_basinnorm.json')
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)
    x_temp = x.copy()
    
    for k in range(len(varLst)):
        var = varLst[k]

        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if (
                    var == 'prcp(mm/day)'
                    or var == '00060_Mean'
                    or var == 'combine_discharge'
                ):
                    x_temp[:, :, k] = np.log10(np.sqrt(x_temp[:, :, k]) + 0.1)

                out[:, :, k] = (x_temp[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if (
                    var == 'prcp(mm/day)'
                    or var == '00060_Mean'
                    or var == 'combine_discharge'
                ):
                    x_temp[:, k] = np.log10(np.sqrt(x_temp[:, k]) + 0.1)
                out[:, k] = (x_temp[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x_temp[:, :, k] * stat[3] + stat[2]
                if (
                    var == 'prcp(mm/day)'
                    or var == '00060_Mean'
                    or var == 'combine_discharge'
                ):
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2

            elif len(x.shape) == 2:
                out[:, k] = x_temp[:, k] * stat[3] + stat[2]
                if (
                    var == 'prcp(mm/day)'
                    or var == '00060_Mean'
                    or var == 'combine_discharge'
                ):
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out


def init_norm_stats(config, x_NN, c_NN, y) -> None:
    stats_directory = config['output_dir']
    statFile = os.path.join(stats_directory, 'statistics_basinnorm.json')

    if not os.path.isfile(statFile):
        # Read all data in training for just the inputs used in NN.
        # Calculate the stats
        calculate_statistics_all(config, x_NN, c_NN, y)
    # with open(statFile, 'r') as fp:
    #     statDict = json.load(fp)
