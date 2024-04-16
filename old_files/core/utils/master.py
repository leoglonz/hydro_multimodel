"""
General utility scripts are kept here.
"""
import json
import os
from collections import OrderedDict

import torch
from core.utils.time import tRange2Array


def readMasterFile(out):
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'r') as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + mFile)
    return mDict


def loadModel(checkpoint, epoch=None, modelName='model'):
    # `checkpoint` is the path to the folder containing the saved model.
    if epoch is None:
        mDict = readMasterFile(checkpoint)
        epoch = mDict['train']['nEpoch']
    modelFile = os.path.join(checkpoint, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model


def create_output_dirs(args):
    seed = args["randomseed"][0]
    # checking rho value first
    t = tRange2Array(args["t_train"])
    if t.shape[0] < args["rho"]:
        args["rho"] = t.shape[0]

    # checking the directory
    if not os.path.exists(args["output_model"]):
        os.makedirs(args["output_model"])
    if args["hydro_model_name"]!= "None":
        hydro_name = args["hydro_model_name"]
    else:
        hydro_name = ""
    if args["temp_model_name"]!= "None":
        temp_name = "_" + args["temp_model_name"]
    else:
        temp_name = ""

    out_folder = args["NN_model_name"] + \
            "_" + hydro_name + \
            temp_name + \
            '_E' + str(args['EPOCHS']) + \
             '_R' + str(args['rho']) + \
             '_B' + str(args['batch_size']) + \
             '_H' + str(args['hidden_size']) + \
             "_tr" + str(args["t_train"][0])[:4] + "_" + str(args["t_train"][1])[:4] + \
            "_n" + str(args["nmul"]) + \
            "_" + str(seed)

    if not os.path.exists(os.path.join(args["output_model"], hydro_name, out_folder)):
        os.makedirs(os.path.join(args["output_model"], hydro_name, out_folder))

    testing_dir = "ts" + str(args["t_test"][0])[:4] + "_" + str(args["t_test"][1])[:4]
    if not os.path.exists(os.path.join(args["output_model"], hydro_name, out_folder, testing_dir)):
        os.makedirs(os.path.join(args["output_model"], hydro_name, out_folder, testing_dir))
    # else:
    #     shutil.rmtree(os.path.join(args['output']['model'], out_folder))
    #     os.makedirs(os.path.join(args['output']['model'], out_folder))
    args["out_dir"] = os.path.join(args["output_model"], hydro_name, out_folder)
    args["testing_dir"] = testing_dir

    # saving the args file in output directory
    config_file = json.dumps(args)
    config_path = os.path.join(args["out_dir"], "config_file.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, "w")
    f.write(config_file)
    f.close()

    return args


def update_args(args, **kw):
    for key in kw:
        if key in args:
            try:
                args[key] = kw[key]
            except ValueError:
                print("Something went wrong in args when updating " + key)
        else:
            print("didn't find " + key + " in args")
    return args