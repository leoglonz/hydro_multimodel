"""
Code to support the processing of CONUS files in non-feather formats until full
support is offered in the PMI.

Adapted from Yalan Song 2024.
"""
import numpy as np
import pandas as pd
import json



def load_conus(config, data_dir, attr_path, shape_id_path, gage_info_path):
    ## TODO: replace pd implementation with config date ranges:
    time = pd.date_range('1980-10-01',f'1995-09-30', freq='d')

    with open(data_folder + 'train_data_dict.json') as f:
        train_data_dict = json.load(f)

    full_period = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
    ind_start = full_period.get_loc(time[0])
    ind_end = full_period.get_loc(time[-1])


    # Load shape id list, target streamflow observations, attributes, forcings,
    # and merit gage info.
    shape_id_lst = np.load(shape_id_path)
    streamflow = np.load(data_dir + 'train_flow.npy')

    all_attributes = pd.read_csv(attr_path,index_col=0)
    all_attributes = all_attributes.sort_values(by='id')

    merit_gage_info = pd.read_csv(gage_info_path) 
    merit_gage_info = merit_gage_info.sort_values(by='STAID')
    gage_ids = merit_gage_info['STAID'].values    

    # Only keep data for gages we want.
    all_attributes = all_attributes[all_attributes['id'].isin(gage_ids)]
    all_attributes_list = all_attributes.columns


    ## Get basin area for unit conversions.
    basin_area = np.expand_dims(all_attributes["area"].values,axis = 1)

    lat =  all_attributes["lat"].values
    id_list_new = all_attributes["id"].values
    id_list_old = [int(id) for id in shape_id_lst]
    [c, ind_1, sub_ind] = np.intersect1d(id_list_new, id_list_old, return_indices=True)

    # Only keep data for gages we want.
    streamflow = streamflow[sub_ind,:,:]
    if(not (id_list_new==np.array(id_list_old)[sub_ind]).all()):
        raise Exception("Ids of subset gage do not match with ids in the attribute file.")






if __name__ == '__main__':
    ### Debug
    config = []
    data_folder = '/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/'
    shape_ids = '/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/shapeID_str_lst.npy'
    attributes = '/data/yxs275/CONUS_data/attributes/CONUS_3254/attributes_haoyu/attributes_haoyu.csv'
    gage_info = '/data/yxs275/CONUS_data/CONUS_for_NH_training/gages_3000_merit_info.csv'
    ###


    load_conus(
        config,
        data_folder,
        attributes,
        shape_ids,
        gage_info
    )
