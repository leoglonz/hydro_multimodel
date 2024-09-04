import json
import os
from typing import Dict, List, Any

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm



def calc_stat_basinnorm(y: np.ndarray, c: np.ndarray, config: Dict[str, Any]) -> List[float]:
    """
    Taken from the calStatsbasinnorm function of hydroDL.
    For daily streamflow normalized by basin area and precipitation.

    :param y: streamflow data to be normalized
    :param c: forcing + attr numpy matrix
    :param config: config file
    :return: statistics to be used for flow normalization
    """
    y[y == (-999)] = np.nan
    y[y < 0] = 0
    attr_list = config['observations']['var_c_nn']

    if 'DRAIN_SQKM' in attr_list:
        area_name = 'DRAIN_SQKM'
    elif 'area_gages2' in attr_list:
        area_name = 'area_gages2'
    else:
        raise ValueError("Unsupported area name.")
    basinarea = c[:, attr_list.index(area_name)]

    if 'PPTAVG_BASIN' in attr_list:
        p_mean_name = 'PPTAVG_BASIN'
    elif 'p_mean' in attr_list:
        p_mean_name = 'p_mean'
    else:
        raise ValueError("Unsupported p_mean name.")
    meanprep = c[:, attr_list.index(p_mean_name)]

    temparea = np.repeat(np.expand_dims(basinarea, axis=(1,2)), y.shape[0]).reshape(y.shape)
    tempprep = np.repeat(np.expand_dims(meanprep, axis=(1, 2)), y.shape[0]).reshape(y.shape)
    
    flowua = (y * 0.0283168 * 3600 * 24) / (
        (temparea * 10**6) * (tempprep * 10**-2) / 365
    )  # unit (m^3/day)/(m^3/day)

    # Apply tranformation to change gamma characteristics, add 0.1 for 0 values.
    a = flowua.flatten()
    b = np.log10( np.sqrt(a[~np.isnan(a)]) + 0.1)

    p10, p90 = np.percentile(b, [10,90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    # if std < 0.001: std = 1

    return [p10, p90, mean, max(std, 0.001)]


def calculate_statistics(x: np.ndarray) -> List[float]:
    """
    Calculate basic statistics excluding NaNs and specific invalid values.
    
    :param x: Input data
    :return: List of statistics [10th percentile, 90th percentile, mean, std]
    """
    a = x.flatten()
    b = a[(~np.isnan(a)) & (a != -999999)]
    p10, p90 = np.percentile(b, [10,90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    # if std < 0.001: std = 1

    return [p10, p90, mean, max(std, 0.001)]


# TODO: Eventually replace calculate_statistics with the version below.
# def calculate_statistics_dmc(data: xr.Dataset, column: str = "time", row: str = "gage_id") -> Dict[str, torch.Tensor]:
#     """
#     Calculate statistics for the data in a similar manner to calStat from hydroDL.

#     :param data: xarray Dataset
#     :param column: Name of the column for calculations
#     :param row: Name of the row for calculations
#     :return: Dictionary with statistics
#     """
#     statistics = {}
#     p_10 = data.quantile(0.1, dim=column)
#     p_90 = data.quantile(0.9, dim=column)
#     mean = data.mean(dim=column)
#     std = data.std(dim=column)
#     col_names = data[row].values.tolist()
#     for idx, col in enumerate(
#         tqdm(col_names, desc="\rCalculating statistics", ncols=140, ascii=True)
#     ):
#         col_str = str(col)
#         statistics[col_str] = torch.tensor(
#             data=[
#                 p_10.streamflow.values[idx],
#                 p_90.streamflow.values[idx],
#                 mean.streamflow.values[idx],
#                 std.streamflow.values[idx],
#             ]
#         )
#     return statistics


def calculate_statistics_gamma(x: np.ndarray) -> List[float]:
    """
    Taken from the cal_stat_gamma function of hydroDL.

    Calculate gamma statistics for streamflow and precipitation data.

    :param x: Input data
    :return: List of statistics [10th percentile, 90th percentile, mean, std]
    """
    a = x.flatten()
    b = a[(~np.isnan(a)) & (a != -999999)]
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # Some tranformation to change gamma characteristics, add 0.1 for 0 values.

    p10, p90 = np.percentile(b, [10,90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    # if std < 0.001: std = 1

    return [p10, p90, mean, max(std, 0.001)]


def calculate_statistics_all(config: Dict[str, Any], x: np.ndarray, c: np.ndarray, y: np.ndarray) -> None:
    """
    Taken from the calStatAll function of hydroDL.
    
    Calculate and save statistics for all variables in the config.

    :param config: Configuration dictionary
    :param x: Forcing data
    :param c: Attribute data
    :param y: Target data
    """
    stat_dict = {}

    # Target stats
    for i, target_name in enumerate(config['target']):
        if target_name == '00060_Mean':
            stat_dict[config['target'][i]] = calc_stat_basinnorm(y[:, :, i:i+1], c, config)
        else:
            stat_dict[config['target'][i]] = calculate_statistics(y[:, :, i:i+1])

    # Forcing stats
    var_list = config['observations']['var_t_nn']
    for k, var in enumerate(var_list):
        if var in config['use_log_norm']:
            stat_dict[var] = calculate_statistics_gamma(x[:, :, k])
        elif var in ['00060_Mean', 'combine_discharge']:
            stat_dict[var] = calc_stat_basinnorm(x[:, :, k: k + 1], x, config)
        else:
            stat_dict[var] = calculate_statistics(x[:, :, k])

    # Attribute stats
    varList = config['observations']['var_c_nn']
    for k, var in enumerate(varList):
        stat_dict[var] = calculate_statistics(c[:, k])

    # Save all stats.
    stat_file = os.path.join(config['output_dir'], 'statistics_basinnorm.json')
    with open(stat_file, 'w') as f:
        json.dump(stat_dict, f, indent=4)


def trans_norm(config: Dict[str, Any], x: np.ndarray, var_lst: List[str], *, to_norm: bool) -> np.ndarray:
    """
    Taken from the trans_norm function of hydroDL.
    
    Transform normalization for the given data.

    :param config: Configuration dictionary
    :param x: Input data
    :param var_lst: List of variables
    :param to_norm: Whether to normalize or de-normalize
    :return: Transformed data
    """
    stat_file = os.path.join(config['output_dir'], 'statistics_basinnorm.json')
    with open(stat_file, 'r') as f:
        stat_dict = json.load(f)

    var_lst = [var_lst] if isinstance(var_lst, str) else var_lst
    out = np.zeros(x.shape)
    x_temp = x.copy()
    
    for k, var in enumerate(var_lst):
        stat = stat_dict[var]

        if to_norm:
            if len(x.shape) == 3:
                if var in config['use_log_norm']: # 'prcp(mm/day)', '00060_Mean', 'combine_discharge
                    x_temp[:, :, k] = np.log10(np.sqrt(x_temp[:, :, k]) + 0.1)
                out[:, :, k] = (x_temp[:, :, k] - stat[2]) / stat[3]

            elif len(x.shape) == 2:
                if var in config['use_log_norm']:
                    x_temp[:, k] = np.log10(np.sqrt(x_temp[:, k]) + 0.1)
                out[:, k] = (x_temp[:, k] - stat[2]) / stat[3]
            else:
                raise ValueError("Incorrect input dimensions. x array must have 2 or 3 dimensions.")
        
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x_temp[:, :, k] * stat[3] + stat[2]
                if var in config['use_log_norm']:
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x_temp[:, k] * stat[3] + stat[2]
                if var in config['use_log_norm']:
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
            else:
                raise ValueError("Incorrect input dimensions. x array must have 2 or 3 dimensions.")

    return out


def init_norm_stats(config: Dict[str, Any], x_NN: np.ndarray, c_NN: np.ndarray, y: np.ndarray) -> None:
    """
    Initialize normalization statistics and save them to a file.

    :param config: Configuration dictionary
    :param x_NN: Neural network input data
    :param c_NN: Attribute data
    :param y: Target data
    """
    stats_directory = config['output_dir']
    stat_file = os.path.join(stats_directory, 'statistics_basinnorm.json')

    if not os.path.isfile(stat_file):
        calculate_statistics_all(config, x_NN, c_NN, y)
        