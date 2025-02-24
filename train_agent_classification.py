from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
import ray

@ray.remote(num_gpus=1, resources={"gpu_lvl_2" : 1})
class Active_Learning_train:
    def __init__(self,   config, 
                         dataset,
                         num_run=0,
                         resume_model_path=False,
                         resume = False):

        
        #############################################################################################
        # LIBRARIES
        #############################################################################################        
        import os
        import numpy as np
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.keras import optimizers, losses, models, backend, layers, metrics
        from tensorflow.keras.utils import multi_gpu_model
        
        self.run_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.run_path)
        
        utils         = local_module("utils")
        logger        = local_module("logger")
        lossnet       = local_module("lossnet")
        data_pipeline = local_module("data_pipeline")
        backbones     = local_module("backbones")
        
        #############################################################################################
        # PARAMETERS RUN
        #############################################################################################
        self.config          = config
        self.dataset         = dataset
        self.num_run         = num_run
        self.group           = "Stage_"+str(num_run)
        self.name_run        = "Train_"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group)
        self.run_dir_check   = os.path.join(self.run_dir ,'checkpoints')
        self.checkpoints_path= os.path.join(self.run_dir_check,'checkpoint.{epoch:03d}.hdf5')
        self.user            = get_user()
        self.training_thread = None
        self.resume_training = resume
        
        #self.num_data_train  = len(labeled_set) 
        self.resume_model_path = resume_model_path
        self.transfer_weight_path = self.config['TRAIN']["transfer_weight_path"]
        self.input_shape       = [self.config["NETWORK"]["INPUT_SIZE"], self.config["NETWORK"]["INPUT_SIZE"], 3]
        
        
        self.pre ='\033[1;36m' + self.name_run + '\033[0;0m' #"____" #
        self.problem ='\033[1;31m' + self.name_run + '\033[0;0m'
        
        # Creating the train folde
        import shutil
        
        # create base dir and gr
        if os.path.exists(config["PROJECT"]["project_dir"]) is False:
            os.mkdir(config["PROJECT"]["project_dir"])
        
        if os.path.exists(self.run_dir) and self.resume_model_path is False:
            shutil.rmtree(config["PROJECT"]["group_dir"])
            os.mkdir(config["PROJECT"]["group_dir"])
            
        if os.path.exists(config["PROJECT"]["group_dir"]) is False:
            os.mkdir(config["PROJECT"]["group_dir"])

        if os.path.exists(self.run_dir) is False:
            os.mkdir(self.run_dir)
            
        if os.path.exists(self.run_dir_check) is False:
            os.mkdir(self.run_dir_check)
            
        
        #############################################################################################
        # SETUP TENSORFLOW SESSION
        #############################################################################################
        # Create a MirroredStrategy.
        #self.strategy = tf.distribute.MirroredStrategy()
        #print(self.pre,'Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        #with self.strategy.scope():
        if True:
            self.graph = tf.Graph()
            with self.graph.as_default():
                config_tf = tf.ConfigProto(allow_soft_placement=True) 
                config_tf.gpu_options.allow_growth = True 
                self.sess = tf.Session(config=config_tf,graph=self.graph)
                backend.set_session(self.sess)
                with self.sess.as_default():

                    #############################################################################################
                    # SETUP WANDB
                    #############################################################################################
                    import wandb

                    self.wandb = wandb
                    self.wandb.init(project  = config["PROJECT"]["project"], 
                                    group    = config["PROJECT"]["group"], 
                                    name     = "Train_"+str(num_run),
                                    job_type = self.group ,
                                    sync_tensorboard = True,
                                    config = config)

                    #############################################################################################
                    # LOAD DATA
                    #############################################################################################

                    self.DataGen = data_pipeline.ClassificationDataset(   config["TRAIN"]["batch_size"],
                                                                    self.dataset,
                                                                    subset = "train",
                                                                    original_size      = config["DATASET"]["original_size"],
                                                                    data_augmentation  = config["DATASET"]["Data_augementation"],
                                                                    random_flip        = config["DATASET"]["random_flip"],
                                                                    pad                = config["DATASET"]["pad"],
                                                                    random_crop_pad    = config["DATASET"]["random_crop_pad"],
                                                                    random_hue         = config["DATASET"]["random_hue"],
                                                                    random_brightness  = config["DATASET"]["random_brightness"],
                                                                    random_saturation  = config["DATASET"]["random_saturation"])  
                    

                    self.num_class = len(self.DataGen.list_classes)

                    #############################################################################################
                    # GLOBAL PROGRESS
                    #############################################################################################
                    self.steps_per_epoch  = int(np.ceil(self.DataGen.nb_elements/config["TRAIN"]["batch_size"]))
                    self.split_epoch   = self.config['TRAIN']["EPOCH_WHOLE"] 
                    self.total_epochs  = self.config['TRAIN']["EPOCH_WHOLE"] + self.config['TRAIN']["EPOCH_SLIT"]
                    self.total_steps   = self.steps_per_epoch*self.total_epochs
                    
                    #############################################################################################
                    # DEFINE CLASSIFIER
                    #############################################################################################
                    # set input
                    img_input = tf.keras.Input(tensor=self.DataGen.images_tensor,name= 'input_image')
                    #img_input = tf.keras.Input(self.input_shape,name= 'input_image')

                    include_top = True

                    # Get the selected backbone
                    """
                    ResNet18
                    ResNet50
                    ResNet101
                    ResNet152
                    ResNet50V2
                    ResNet101V2
                    ResNet152V2
                    ResNeXt50
                    ResNeXt101
                    """
                    print(self.pre, "The backbone is: ",self.config["NETWORK"]["Backbone"])
                    self.backbone = getattr(backbones,self.config["NETWORK"]["Backbone"])
                    #
                    c_pred_features = self.backbone(input_tensor=img_input, classes= self.num_class, include_top=include_top)
                    self.c_pred_features=  c_pred_features
                    if include_top: # include top classifier
                        # class predictions
                        c_pred = c_pred_features[0]
                    else:
                        x = layers.GlobalAveragePooling2D(name='pool1')(c_pred_features[0])
                        x = layers.Dense(self.num_class, name='fc1')(x)
                        c_pred = layers.Activation('softmax', name='c_pred')(x)
                        c_pred_features[0] = c_pred

                    #self.classifier = models.Model(inputs=[img_input], outputs=c_pred_features,name='Classifier') 


                    #############################################################################################
                    # DEFINE FULL MODEL
                    #############################################################################################
                    #c_pred_features_1 = self.classifier(img_input)
                    #c_pred_1 = c_pred_features[0]
                    loss_pred_embeddings = lossnet.Lossnet(c_pred_features, self.config["NETWORK"]["embedding_size"])
                                        
                    model_inputs  = [img_input]
                    model_outputs = [c_pred] + loss_pred_embeddings

                    self.model = models.Model(inputs=model_inputs, outputs=model_outputs) #, embedding_s] )
                    
                    ########################################
                    # INIT GLOBAL VARIABLES
                    #######################################
                    self.sess.run(tf.global_variables_initializer())
                    
                    #############################################################################################
                    # LOAD PREVIUS WEIGTHS
                    #############################################################################################
                    if self.resume_model_path:
                        # check the epoch where is loaded
                        try:
                            loaded_epoch = int(self.resume_model_path.split('.')[-2])
                            print(self.pre, "Loading weigths from: ",self.resume_model_path)
                            print(self.pre, "The detected epoch is: ",loaded_epoch)
                            # load weigths
                            self.model.load_weights(self.resume_model_path)
                        except:
                            print( self.problem ,"=> Problem loading the weights from ",self.resume_model_path)
                            print( self.problem ,'=> It will rain from scratch')
                    elif self.transfer_weight_path:
                        try:
                            print(self.pre, "(transfer learning) Loading weigths by name from: ",self.transfer_weight_path)
                            # load weigths
                            self.model.load_weights(self.transfer_weight_path, by_name=True )
                        except:
                            print( self.problem ,"=>(transfer learning) Problem loading the weights from ",self.transfer_weight_path)
                            print( self.problem ,'=> It will rain from scratch')

                            
                    if self.resume_training:
                        self.current_epoch = loaded_epoch
                        self.current_step  = loaded_epoch*self.steps_per_epoch

                        if self.current_epoch > self.total_epochs:
                            raise ValueError("The starting epoch is higher that the total epochs")
                        else:
                            print(self.pre, "Resuming the training from stage: ",self.num_run," at epoch ", self.current_epoch)
                    else:
                        self.current_epoch = 0
                        self.current_step  = 0

                    #############################################################################################
                    # DEFINE WEIGHT DECAY
                    #############################################################################################
                    if self.config['TRAIN']['apply_weight_decay']:
                        utils.add_weight_decay(self.model,self.config['TRAIN']['weight_decay'])

                    #############################################################################################
                    # DEFINE LOSSES
                    #############################################################################################
                    
                    # losses
                    self.loss_dict = {}
                    self.loss_dict['c_pred']    = losses.sparse_categorical_crossentropy
                    self.loss_dict['l_pred_w']  = lossnet.Loss_Lossnet
                    self.loss_dict['l_pred_s']  = lossnet.Loss_Lossnet
                    # weights
                    self.weight_w = backend.variable(self.config['TRAIN']['weight_lossnet_loss'])
                    self.weight_s = backend.variable(0)

                    self.loss_w_dict = {}
                    self.loss_w_dict['c_pred']   = 1.0
                    self.loss_w_dict['l_pred_w'] = self.weight_w
                    self.loss_w_dict['l_pred_s'] = self.weight_s
                    #self.loss_w_dict['Embedding']  = 0

                    #############################################################################################
                    # DEFINE METRICS
                    #############################################################################################
                    # metrics
                    self.metrics_dict = {}
                    self.metrics_dict['c_pred']      = tf.keras.metrics.SparseCategoricalAccuracy()
                    #self.metrics_dict['l_pred_w']   = lossnet.MAE_Lossnet
                    #self.metrics_dict['l_pred_s']   = lossnet.MAE_Lossnet

                    #############################################################################################
                    # DEFINE OPTIMIZER
                    #############################################################################################
                    self.opt = optimizers.Adam(lr=self.config['TRAIN']['lr'])

                    #############################################################################################
                    # DEFINE CALLBACKS
                    #############################################################################################
                    # Checkpoint saver
                    self.callbacks = []
                    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                            filepath=self.checkpoints_path,
                                                            save_weights_only=True,
                                                            period=self.config["TRAIN"]["test_each"])


                    self.callbacks.append(model_checkpoint_callback)

                    # Callback to wandb
                    # i remplaced this for a custom callback that saves the logs with better names
                    #self.callbacks.append(self.wandb.keras.WandbCallback())

                    # Callback Learning Rate
                    def scheduler(epoch):
                        lr = self.config['TRAIN']['lr']
                        for i in self.config['TRAIN']['MILESTONES']:
                            if epoch>i:
                                lr*=0.1
                        return lr

                    #self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
                    
                    # callback to change the weigths for the split training:
                    self.callbacks.append(lossnet.Change_loss_weights(self.weight_w, 
                                                                      self.weight_s, 
                                                                      self.split_epoch, 
                                                                      self.config['TRAIN']['weight_lossnet_loss']))
                    
                    ##################
                    # SETUP WATCHER
                    ################## 
                    self.run_watcher = get_run_watcher()
                    
                    self.run_watcher.add_run.remote(name=self.name_run,
                                                    user=self.user,
                                                    progress=0,
                                                    wandb_url=self.wandb.run.get_url(),
                                                    status="Idle")

                    # Callback update progress
                    self.Update_progress = logger.Update_progress(  self.run_watcher,
                                                                    self.wandb,
                                                                    self.name_run,
                                                                    self.steps_per_epoch, 
                                                                    self.total_epochs, 
                                                                    self.total_steps, 
                                                                    self.current_epoch, 
                                                                    self.current_step)
                    
                    self.callbacks.append(self.Update_progress)

                    #############################################################################################
                    # COMPILE MODEL
                    #############################################################################################        
                    self.model.compile(loss = self.loss_dict, 
                                       loss_weights = self.loss_w_dict, 
                                       metrics = self.metrics_dict, 
                                       optimizer = self.opt,
                                       target_tensors=self.DataGen.labels_tensor)

                    ########################################
                    # INIT LOCAL VARIABLES
                    #######################################
                    self.sess.run(tf.local_variables_initializer())

            print(self.pre,'Init done')

    @ray.method(num_returns = 0)
    def start_training(self):
        import threading
        import os
        import numpy as np
        from copy import deepcopy
        from tensorflow.keras import backend
        import tensorflow as tf
        
        def train():
            try:
                #with self.strategy.scope():
                if True:
                    with self.graph.as_default():
                        with self.sess.as_default():
                            print( self.pre ,"Start training")
                            
                            ###############################################################################
                            # TRAIN THE WHOLE NETWORK
                            ###############################################################################
                            if self.current_epoch > self.total_epochs:
                                print(self.problem, 'The starting epoch is higher that the total epochs')
                            else:
                                history = self.model.fit(  steps_per_epoch = self.steps_per_epoch,
                                                           epochs = self.total_epochs, 
                                                           callbacks = self.callbacks,
                                                           initial_epoch = self.current_epoch,
                                                           verbose=2)
                
            except Exception as e:
                self.run_watcher.update_run.remote(name=self.name_run, status="Error")
                print(self.problem ,e)
            
        if self.training_thread is None or not self.training_thread.is_alive():
            self.Update_progress.stop_flag = False
            self.training_thread = threading.Thread(target=train, args=(), daemon=True)
            self.training_thread.start()
            
    @ray.method(num_returns = 1)
    def isTraining(self):
        return not (self.training_thread is None or not self.training_thread.is_alive())

    @ray.method(num_returns = 0)
    def stop_training(self):
        self.Update_progress.stop_flag = True

    @ray.method(num_returns = 1)
    def get_progress(self):
        return self.Update_progress.status