import ray
ray.init(address="auto", ignore_reinit_error=True)

# load defaulft config
import yaml
import os
import time
import pandas as pd
import random

config_path = './configs/Active_v2.yml'

with open(config_path) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)
    

# create base dir and gr
if os.path.exists(config["PROJECT"]["project_dir"]) is False:
    os.mkdir(config["PROJECT"]["project_dir"])

if os.path.exists(config["PROJECT"]["group_dir"]) is False:
    os.mkdir(config["PROJECT"]["group_dir"])
    
    
# Get the data to annotate

#############################################################################################
# LOAD DATA
#############################################################################################
from data_utils import CIFAR10Data
# Load data
cifar10_data = CIFAR10Data()
num_classes = len(cifar10_data.classes)
x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)

indices = list(range(len(x_train)))
random.seed(101)
random.shuffle(indices)
labeled_set = indices[:config["RUNS"]["ADDENDUM"] ]
unlabeled_set = indices[config["RUNS"]["ADDENDUM"] :]


# test with all the images
NUM_IMAGES_TEST = len(x_test)
# Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
test_set = list(range(NUM_IMAGES_TEST))

config["NETWORK"]["INPUT_SIZE"] =  x_train[0].shape[0]
config["NETWORK"]["CLASSES"] = cifar10_data.classes

print(config)

from train_agent_cifar import Active_Learning_train
from inference_agent_cifar import Active_Learning_inference


for num_run in range(10):
    if num_run==0:
        initial_weight_path = False
    else:
        initial_weight_path = os.path.join(config['PROJECT']['group_dir'],'Stage_'+str(num_run-1),'checkpoint','epoch200.ckpt-200')
        
        
    NetworkActor =  Active_Learning_train.remote(config, labeled_set, test_set,  num_run, initial_weight_path)
    NetworkActor.start_training.remote()
    
    # Wait util the model is training
    while True:
        time.sleep(10)
        try:
            progress_id = NetworkActor.isTraining.remote()
            response = ray.get(progress_id)
            break
        except:
            pass
        
    # wait until the model finish training
    while True:
        time.sleep(10)
        progress_id = NetworkActor.isTraining.remote()
        response = ray.get(progress_id)
        if not response:
            break
    
    NetworkActor.__ray_terminate__.remote()
    
    del NetworkActor

        
    weight_file = os.path.join(config['PROJECT']['group_dir'],'Stage_'+str(num_run),'checkpoint','epoch200.ckpt-200')

    AL_inference = Active_Learning_inference.remote( config, unlabeled_set, num_run, weight_file)
    AL_inference.evaluate.remote()

    run_dir   = os.path.join(config["PROJECT"]["group_dir"],"Stage_"+str(num_run))
    ordered_indexes   = os.path.join(run_dir, "ordered_indexes.csv")

    # wait the file qith the scores is generated
    while True:
        time.sleep(10)
        if os.path.isfile(ordered_indexes):
            break

    # read the scores file and create the new labeled set and unlabeled set to repeat the trainig
    pd_ordered_indexes = pd.read_csv(ordered_indexes)
    new_annotated_data = list(pd_ordered_indexes.iloc[:config["RUNS"]["ADDENDUM"]]['indexes'].to_numpy())
    labeled_set += new_annotated_data
    unlabeled_set =  list(pd_ordered_indexes.iloc[config["RUNS"]["ADDENDUM"]:]['indexes'].to_numpy())

from test_agent_cifar_all import Active_Learning_test_all

NetworkActor =  Active_Learning_test_all.remote(config)
NetworkActor.evaluate.remote()