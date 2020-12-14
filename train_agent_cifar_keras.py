from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
import ray

@ray.remote(num_gpus=1)
class Active_Learning_train:
    def __init__(self,   config, 
                         labeled_set,
                         test_set, 
                         num_run,
                         resume_model_path,
                         resume = False):

        
        #############################################################################################
        # LIBRARIES
        #############################################################################################        
        import os
        self.run_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.run_path)
        
        import tensorflow as tf
        
        core = local_module("core")

        backbones = local_module("backbones")
        
        from tensorflow.python import pywrap_tensorflow
        import numpy as np
        from tensorflow.keras import optimizers, losses, models, backend, layers, metrics
        
        #############################################################################################
        # PARAMETERS RUN
        #############################################################################################
        self.config          = config
        self.num_run         = num_run
        self.group           = "Stage_"+str(num_run)
        self.name_run        = "Train_"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group)
        self.run_dir_check   = os.path.join(self.run_dir ,'checkpoints')
        self.checkpoints_path= os.path.join(self.run_dir_check,'checkpoint.{epoch:03d}.hdf5')
        self.user            = get_user()
        self.test_run_id     = None
        self.stop_flag       = False
        self.training_thread = None
        self.resume_training = resume
        
        self.num_data_train  = len(labeled_set) 
        self.resume_model_path = resume_model_path
        self.transfer_weight_path = self.config['TRAIN']["transfer_weight_path"]
        self.num_class       = len(self.config["NETWORK"]["CLASSES"])
        self.input_shape     = [self.config["NETWORK"]["INPUT_SIZE"], self.config["NETWORK"]["INPUT_SIZE"], 3]
        
        
        self.pre ='\033[1;36m' + self.name_run + '\033[0;0m' #"____" #
        self.problem ='\033[1;31m' + self.name_run + '\033[0;0m'
        
        # Creating the train folde
        import shutil
        
        if os.path.exists(self.run_dir) and self.resume_model_path is False:
            if num_run==0:
                shutil.rmtree(config["PROJECT"]["group_dir"])
                os.mkdir(config["PROJECT"]["group_dir"])
            else:  
                shutil.rmtree(self.run_dir)
                
        if os.path.exists(self.run_dir) is False:
            os.mkdir(self.run_dir)
            
        if os.path.exists(self.run_dir_check) is False:
            os.mkdir(self.run_dir_check)
            
        #############################################################################################
        # GLOBAL PROGRESS
        #############################################################################################
        self.current_epoch = 0
        self.split_epoch   = self.config['TRAIN']["EPOCH_WHOLE"] 
        self.total_epochs  = self.config['TRAIN']["EPOCH_WHOLE"] + self.config['TRAIN']["EPOCH_SLIT"]
        self.progress = round(self.current_epoch / self.total_epochs * 100.0, 2)
        
        #############################################################################################
        # SETUP TENSORFLOW SESSION
        #############################################################################################

        self.graph = tf.Graph()
        with self.graph.as_default():
            config_tf = tf.ConfigProto(allow_soft_placement=True) 
            config_tf.gpu_options.allow_growth = True 
            self.sess = tf.Session(config=config_tf,graph=self.graph)
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
                if self.config["PROJECT"]["source"]=='CIFAR':
                    from data_utils import CIFAR10Data
                    # Load data
                    cifar10_data = CIFAR10Data()
                    x_train, y_train, _, _ = cifar10_data.get_data(normalize_data=False)

                    x_train = x_train[labeled_set]
                    y_train = y_train[labeled_set]

                    self.test_set = test_set
                else:
                    raise NameError('This is not implemented yet')


                #############################################################################################
                # DATA GENERATOR
                #############################################################################################
                self.Data_Generator = core.Generator_cifar_train(x_train, y_train, config)

                #############################################################################################
                # GENERATE MODEL
                #############################################################################################

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
                #############################################################################################
                # DEFINE CLASSIFIER
                #############################################################################################
                # set input
                img_input = tf.keras.Input(self.input_shape,name= 'input_image')

                include_top = True

                # Get the selected backbone
                self.backbone = getattr(backbones,"ResNet18_cifar")
                #
                c_pred_features = self.backbone(input_tensor=img_input, classes= self.num_class, include_top=include_top)
                self.c_pred_features= c_pred_features
                if include_top: # include top classifier
                    # class predictions
                    c_pred = c_pred_features[0]
                else:
                    x = layers.GlobalAveragePooling2D(name='pool1')(c_pred_features[0])
                    x = layers.Dense(self.num_class, name='fc1')(x)
                    c_pred = layers.Activation('softmax', name='c_pred')(x)
                    c_pred_features[0]=c_pred

                self.classifier = models.Model(inputs=[img_input], outputs=c_pred_features,name='Classifier') 

                #############################################################################################
                # DEFINE FULL MODEL
                #############################################################################################
                c_pred_features_1 = self.classifier(img_input)
                c_pred_1 = c_pred_features_1[0]

                # define lossnet
                loss_pred_embeddings = core.Lossnet(c_pred_features_1, self.config["NETWORK"]["embedding_size"])

                self.model = models.Model(inputs=img_input, outputs=[c_pred_1]+loss_pred_embeddings) #, embedding_s] )

                #############################################################################################
                # DEFINE LOSSES
                #############################################################################################
                # losses
                self.loss_dict = {}
                self.loss_dict['Classifier'] = losses.categorical_crossentropy
                self.loss_dict['l_pred_w']   = core.Loss_Lossnet
                self.loss_dict['l_pred_s']   = core.Loss_Lossnet
                # weights
                self.weight_w = backend.variable(1)
                self.weight_s = backend.variable(0)

                self.loss_w_dict = {}
                self.loss_w_dict['Classifier'] = 1
                self.loss_w_dict['l_pred_w']   = self.weight_w
                self.loss_w_dict['l_pred_s']   = self.weight_s
                self.loss_w_dict['Embedding']  = 0

                #############################################################################################
                # DEFINE METRICS
                #############################################################################################
                # metrics
                self.metrics_dict = {}
                self.metrics_dict['Classifier'] = metrics.categorical_accuracy
                self.metrics_dict['l_pred_w']   = core.MAE_Lossnet
                self.metrics_dict['l_pred_s']   = core.MAE_Lossnet

                #############################################################################################
                # DEFINE OPTIMIZER
                #############################################################################################
                self.opt = optimizers.Adam(lr=0.01)

                #############################################################################################
                # DEFINE CALLBACKS
                #############################################################################################
                # Checkpoint saver
                self.callbacks = []
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                        filepath=self.checkpoints_path,
                                                        save_weights_only=True,
                                                        period=self.config["RUNS"]["test_each"])


                self.callbacks.append(model_checkpoint_callback)

                # Callback to wandb
                self.callbacks.append(self.wandb.keras.WandbCallback())

                # Callback Learning Rate
                def scheduler(epoch):
                    lr = self.config['TRAIN']['lr']
                    for i in self.config['TRAIN']['MILESTONES']:
                        if epoch>i:
                            lr*=0.1
                    return lr

                self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

                # callbeck to change the weigths for the split training:
                self.callbacks.append(core.Change_loss_weights(self.weight_w, self.weight_s, self.split_epoch))

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

                if self.resume_training:
                    self.current_epoch = loaded_epoch
                    self.progress = round(self.current_epoch / self.total_epochs * 100.0, 2)

                    if self.current_epoch > self.total_epochs:
                        raise ValueError("The starting epoch is higher that the total epochs")
                    else:
                        print(self.pre, "Resuming the training from stage: ",self.num_run," at epoch ", self.current_epoch)

                #############################################################################################
                # COMPILE MODEL
                #############################################################################################        
                self.model.compile(loss = self.loss_dict, 
                                   loss_weights = self.loss_w_dict, 
                                   metrics = self.metrics_dict, 
                                  optimizer = self.opt)

                #############################################################################################
                # INIT VARIABLES
                #############################################################################################
                #self.sess.graph.as_default()
                backend.set_session(self.sess)
                self.sess.run(tf.local_variables_initializer())

                #############################################################################################
                # SETUP WATCHER
                #############################################################################################    

                self.run_watcher = get_run_watcher()

                self.run_watcher.add_run.remote(name=self.name_run,
                                                user=self.user,
                                                progress=self.progress,
                                                wandb_url=self.wandb.run.get_url(),
                                                status="Idle")
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
                with self.graph.as_default():
                    with self.sess.as_default():
                        print( self.pre ,"Start training")
                        self.run_watcher.update_run.remote(name=self.name_run, status="Training")

                        ###############################################################################
                        # TRAIN THE WHOLE NETWORK
                        ###############################################################################
                        if self.current_epoch > self.total_epochs:
                            print(self.problem, 'The starting epoch is higher that the total epochs')

                        for epoch in range(self.current_epoch, self.total_epochs+1):

                            if self.stop_flag:
                                self.run_watcher.update_run.remote(name=self.name_run, status="Idle")
                                break

                            history = self.model.fit_generator(self.Data_Generator,
                                                               epochs=epoch+1, 
                                                               callbacks = self.callbacks,
                                                               initial_epoch=epoch,
                                                               verbose=2)

                            self.current_epoch = epoch
                            self.progress = round(self.current_epoch / self.total_epochs * 100.0, 2)
                            self.run_watcher.update_run.remote(name=self.name_run, progress=self.progress)

                self.run_watcher.update_run.remote(name=self.name_run, status="Finished", progress=self.progress)
                
            except Exception as e:
                self.run_watcher.update_run.remote(name=self.name_run, status="Error")
                print( self.problem ,e)
            
        if self.training_thread is None or not self.training_thread.is_alive():
            self.stop_flag=False
            self.training_thread = threading.Thread(target=train, args=(), daemon=True)
            self.training_thread.start()
            
    @ray.method(num_returns = 1)
    def isTraining(self):
        return not (self.training_thread is None or not self.training_thread.is_alive())

    @ray.method(num_returns = 0)
    def stop_training(self):
        self.stop_flag=True

    @ray.method(num_returns = 1)
    def get_progress(self):
        return {"global_step" : self.current_epoch, "progress": self.progress, }