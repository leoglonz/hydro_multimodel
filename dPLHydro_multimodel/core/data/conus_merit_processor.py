"""
Code to support the processing of CONUS files in non-feather formats until full
support is offered in the PMI.

Adapted from Yalan Song 2024.
"""
import numpy as np
import pandas as pd
import json
import zarr



def load_conus(config):
    ## TODO: replace pd implementation with config date ranges:
    time = pd.date_range('1980-10-01',f'1995-09-30', freq='d')

    with open(config['observations']['data_dir'] + 'train_data_dict.json') as f:
        train_data_dict = json.load(f)

    full_period = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
    ind_start = full_period.get_loc(time[0])
    ind_end = full_period.get_loc(time[-1])


    # Load shape id list, target streamflow observations, attributes, forcings,
    # and merit gage info.
    shape_id_lst = np.load(config['observations']['shape_id_path'], )
    streamflow = np.load(config['observations']['data_dir'] + 'train_flow.npy')

    all_attributes = pd.read_csv(config['observations']['attr_path'],index_col=0)
    all_attributes = all_attributes.sort_values(by='id')

    merit_gage_info = pd.read_csv(config['observations']['gage_info_path']) 
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
    

    forcing_list_water_loss = ['P','Temp','PET']

    attributes = ['ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']


    attributes_water_loss = attributes.copy()
    attributes_water_loss.append('uparea')

    attributes_old = attributes.copy()


    forcing_list_water_loss = ['P','Temp','PET']



    key_info = [str(x) for x in id_list_new]



    with open(config['observations']['area_info_path']) as f:
        area_info = json.load(f)

    merit_save_path = config['observations']['merit_path']

    with open(config['observations']['merit_idx_path']) as f:
        merit_idx = json.load(f)

    
    root_zone = zarr.open_group(save_path_merit_path, mode = 'r')
    Merit_all = root_zone['COMID'][:]
    xTrain2 = np.full((len(Merit_all),len(time),len(forcing_list_water_loss)),np.nan)
    attr2 = np.full((len(Merit_all),len(attributes_water_loss)),np.nan)


    merit_time = pd.date_range('1980-10-01',f'2010-09-30', freq='d')
    merit_start_idx = merit_time.get_loc(time[0])
    merit_end_idx = merit_time.get_loc(time[-1])+1


    for fid, foring_ in enumerate(forcing_list_water_loss):    
        xTrain2[:,:,fid] = root_zone[foring_][:,merit_start_idx:merit_end_idx]
    # load attributes

    for aid, attribute_ in enumerate(attributes_water_loss):                                    
        attr2[:,aid] =  root_zone['attr'][attribute_][:] 

    Ac_all = root_zone['attr']["uparea"][:] 
    Ai_all = root_zone['attr']["catchsize"][:] 


    





def get_data_dict(config, train=False):
    """
    Create dictionary of datasets used by the models.
    Contains 'c_nn', 'obs', 'x_hydro_model', 'c_hydro_model', 'inputs_nn_scaled'.

    train: bool, specifies whether data is for training.
    """
    ### TODO: modify for merit/CONUS
    # Get date range.
    config['train_t_range'] = Dates(config['train'], config['rho']).date_to_int()
    config['test_t_range'] = Dates(config['test'], config['rho']).date_to_int()
    config['t_range'] = [config['train_t_range'][0], config['test_t_range'][1]]

    # Create stats for NN input normalizations.
    if train: 
        dataset_dict = load_data(config, config['train_t_range'])
        init_norm_stats(config, dataset_dict['x_nn'], dataset_dict['c_nn'],
                              dataset_dict['obs'])
    else:
        dataset_dict = load_data(config, config['test_t_range'])

    # Normalization
    x_nn_scaled = trans_norm(config, dataset_dict['x_nn'],
                             var_lst=config['observations']['var_t_nn'], to_norm=True)
    c_nn_scaled = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['observations']['var_c_nn'], to_norm=True)
    c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)

    dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
    del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']

    return dataset_dict, config



if __name__ == '__main__':
    ### Debug
    config = {
        'observations':
            {'data_dir': '/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/',
            'shape_id_path': '/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/shapeID_str_lst.npy',
            'attr_path': '/data/yxs275/CONUS_data/attributes/CONUS_3254/attributes_haoyu/attributes_haoyu.csv',
            'gage_info_path': '/data/yxs275/CONUS_data/CONUS_for_NH_training/gages_3000_merit_info.csv',
            'area_info_path': '/data/lgl5139/data/conus/conus_3200/area_info.json',
            'merit_idx_path': '/data/lgl5139/data/conus/conus_3200/merit_idx.json'
            }
        }
    ###


    load_conus(config)
