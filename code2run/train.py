config_path = './configs/Datanet_12_01_0.yml'

###################################################################
###################################################################
###################################################################
###################################################################
import ray
ray.init(address="auto", ignore_reinit_error=True)

from AutoML import AutoML, DataNet, AutoMLDataset
#DataNet().restart_datanodes()
dataset = AutoMLDataset(name="person_classification")#, semantics=[113988, 113989, 113990], crops=True) #, override=True)

saved_to_check = ['43138978_48512491',
                 '43142767_44712676',
                 '43141386_48311841',
                 '43144739_48414797',
                 '43133071_44724167',
                 '43135734_48299637',
                 '9832634_48481924',
                 '43135919_48430240',
                 '10290317_48336963',
                 '43133380_48318079',
                 '43137703_48476395',
                 '43141686_48301188']

to_check_order = []
for subset in ['test_images', 'train_images']:
    for key in dataset.dataset_header[subset].keys():
        to_check_order.append(dataset.dataset_header[subset][key][0])
        to_check_order.append(dataset.dataset_header[subset][key][-1])
    
assert to_check_order == saved_to_check

# load defaulft config
import yaml


with open(config_path) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)

print(config)


from train_agent_classification import Active_Learning_train

Active_Learning = Active_Learning_train.remote(config, 
                                             dataset,
                                             num_run = -1,
                                             resume_model_path=False,
                                             resume = False)

Active_Learning.start_training.remote()