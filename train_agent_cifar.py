from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
import ray




@ray.remote(num_gpus=1)
class Active_Learning_train:
    def __init__(self,   config, 
                         labeled_set,
                         test_set, 
                         num_run,
                         initial_weight_path):

        
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
        
    
        
        #############################################################################################
        # PARAMETERS RUN
        #############################################################################################
        self.config          = config
        self.num_run         = num_run
        self.group           = "Stage_"+str(num_run)
        self.name_run        = "Train_"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group)
        self.user            = get_user()
        self.test_run_id     = None
        self.stop_flag       = False
        self.training_thread = None
        
        self.num_data_train  = len(labeled_set) 
        self.initial_weight_path = initial_weight_path
        self.transfer_weight_path = self.config['TRAIN']["transfer_weight_path"]
        self.num_class       = len(self.config["NETWORK"]["CLASSES"])
        
        self.pre ='\x1b[6;30;42m' + self.name_run + '\x1b[0m' #"____" #

        
        # Creating the train folde
        import shutil
        
        if os.path.exists(self.run_dir) and self.initial_weight_path is False:
            if num_run==0:
                shutil.rmtree(config["PROJECT"]["group_dir"])
                os.mkdir(config["PROJECT"]["group_dir"])
            else:  
                shutil.rmtree(self.run_dir)
                
        if os.path.exists(self.run_dir) is False:
            os.mkdir(self.run_dir)

            
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
            x_train, y_train, _, _ = cifar10_data.get_data(subtract_mean=False)

            x_train = x_train[labeled_set]
            y_train = y_train[labeled_set]
            
            self.test_set = test_set
        else:
            raise NameError('This is not implemented yet')
        
        
        #############################################################################################
        # DATA GENERATOR
        #############################################################################################
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(
                    width_shift_range=self.config["DATASET"]["width_shift_range"],
                    height_shift_range=self.config["DATASET"]["height_shift_range"],
                    horizontal_flip=self.config["DATASET"]["horizontal_flip"])

        train_gen = train_datagen.flow(x_train,
                                       y_train,
                                       batch_size=self.config["TRAIN"]["batch_size"])
        
        wh = self.config["NETWORK"]["INPUT_SIZE"]
        
        features_shape = [None, wh, wh, 3]
        labels_shape = [None, self.num_class ]
        

        tf_data = tf.data.Dataset.from_generator(lambda: train_gen, 
                                                 output_types=(tf.float32, tf.float32),
                                                output_shapes = (tf.TensorShape(features_shape), tf.TensorShape(labels_shape)))
        #tf_data = tf_data.apply(tf.contrib.data.ignore_errors())
        data_tensors = tf_data.make_one_shot_iterator().get_next()
        
        tf_data = tf_data.repeat()
        self.img_input = data_tensors[0]
        self.c_true    = data_tensors[1]
        
        
        #############################################################################################
        # GENERATE MODEL
        #############################################################################################
        self.trainable = True
        
        # Get the selected backbone
        self.backbone = getattr(backbones,self.config["PROJECT"]["Backbone"])
        
        with tf.compat.v1.variable_scope("Backbone"):
            #ResNet18(classes, input_shape, weight_decay=1e-4)
            c_pred_features = self.backbone(self.img_input, self.num_class, self.trainable)
            self.c_pred = c_pred_features[0]
        
        with tf.compat.v1.variable_scope("LossNet"):
            self.l_pred_w, self.l_pred_s, self.embedding_whole, self.embedding_split = core.Lossnet(c_pred_features, self.config["NETWORK"]["embedding_size"])
            
        with tf.name_scope("Define_loss"):           
            # get the classifier
            self.Losses_compute = core.Loss_Lossnet(margin = self.config["NETWORK"]["MARGIN"])
            self.c_loss_nr, self.c_loss, self.l_loss_w, self.l_loss_s = self.Losses_compute.compute_loss(self.c_true, self.c_pred, self.l_pred_w, self.l_pred_s)
            
            self.t_loss_w =  ((self.config['TRAIN']["w_c_loss"]*self.c_loss) + \
                              (self.config['TRAIN']["w_l_loss"] * self.l_loss_w))
            
            self.t_loss_s =  ((self.config['TRAIN']["w_c_loss"]*self.c_loss) + \
                              (self.config['TRAIN']["w_l_loss"] * self.l_loss_s))
            

            
        #############################################################################################
        # GLOBAL PROGRESS
        #############################################################################################
        self.start_epoch  = self.config['TRAIN']["start_epoch"]
        self.split_epoch  = self.config['TRAIN']["EPOCH_WHOLE"] 
        self.total_epochs = self.config['TRAIN']["EPOCH_WHOLE"] + self.config['TRAIN']["EPOCH_SLIT"]

        self.steps_per_epoch = int(np.ceil(self.num_data_train / self.config['TRAIN']["batch_size"]))
        self.global_step_val = self.steps_per_epoch * self.start_epoch
        self.global_step_goal = self.steps_per_epoch * self.total_epochs
        self.progress = round(self.global_step_val / self.global_step_goal * 100.0, 2)
        self.step_end_epoch = 0
        
        
        #############################################################################################
        # DEFINE LEARNING RATE FOR TRAINING
        #############################################################################################
        with tf.name_scope('learning_rate'):
            # define the current step as tf variable
            self.global_step = tf.Variable(float(self.global_step_val), dtype=tf.float64, trainable=False, name='global_step')
            
            # define warmup_steps in tf
            warmup_steps = tf.constant(self.config['TRAIN']["EPOCH_WARMUP"] * self.steps_per_epoch, \
                                       dtype=tf.float64, name='warmup_steps')
            # define train_steps in tf
            train_steps = tf.constant((self.config['TRAIN']["EPOCH_WHOLE"] + 
                                       self.config['TRAIN']["EPOCH_SLIT"] )*self.steps_per_epoch, \
                                      dtype=tf.float64, name='train_steps')
            
            steps_per_epoch = tf.constant(self.steps_per_epoch, dtype=tf.float64, name='steps_per_epoch')
            
            MILESTONES = tf.constant(self.config['TRAIN']["MILESTONES"], dtype=tf.float64)
            
            self.learning_rate = tf.cond(   pred = self.global_step < warmup_steps,
                                            true_fn=lambda: self.global_step / warmup_steps * self.config['TRAIN']["lr"],
                                            false_fn=lambda: self.config['TRAIN']["lr"] * \
                                                     (self.config['TRAIN']["gamma"] ** \
                                                      (tf.compat.v1.reduce_sum(tf.cast(MILESTONES<(self.global_step/steps_per_epoch), tf.float64)))))
            
            global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)
            
            
        #############################################################################################
        # SET REGULARIZATION 
        #############################################################################################
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.config['TRAIN']["wdecay"]).apply(tf.trainable_variables())
            
        
        #############################################################################################
        # DEFINE THE PARAMETERS TO TRAIN
        #############################################################################################
        self.backbone_trainable = []
        for var in tf.trainable_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            if var_name_mess[0] == 'Backbone':
                self.backbone_trainable.append(var)
                    
        self.lossnet_trainable = []
        for var in tf.trainable_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            if var_name_mess[0] == 'LossNet':
                self.lossnet_trainable.append(var)
        
        with tf.name_scope("define_train_whole"):
            op_w_backbone = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss_w, var_list=self.backbone_trainable)
            op_w_lossnet  = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss_w, var_list=self.lossnet_trainable )
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([op_w_backbone, op_w_lossnet, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_whole = tf.no_op()

        with tf.name_scope("define_train_split"):
            op_s_backbone = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss_s, var_list=self.backbone_trainable)
            op_s_lossnet  = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss_s, var_list=self.lossnet_trainable )
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([op_s_backbone, op_s_lossnet, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_split = tf.no_op()
                    
        
        #############################################################################################
        # METRICS
        ############################################################################################# 
        with tf.name_scope("define_metrics"):
            
            with tf.name_scope('Categorical_Accuracy'):
                correct_prediction = tf.equal( tf.argmax(self.c_true, 1), tf.argmax(self.c_pred, 1))
                self.Categorical_Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
            with tf.name_scope('MAE_learning_loss_whole'):
                self.MAE_whole = tf.reduce_mean(tf.math.abs(tf.math.subtract(self.c_loss_nr, self.l_loss_w)))

            with tf.name_scope('MAE_learning_loss_split'):
                self.MAE_split = tf.reduce_mean(tf.math.abs(tf.math.subtract(self.c_loss_nr, self.l_loss_s)))

        #############################################################################################
        # LOAD PREVIUS TRAINED MODEL
        #############################################################################################
        # initial weight for training

        
        # load weigths for transfer learning from transfer_weight_path
        with tf.name_scope('loader_and_saver'):
            if self.transfer_weight_path is not False and self.start_epoch == 0 and False:
                try:
                    reader = pywrap_tensorflow.NewCheckpointReader(self.transfer_weight_path)
                    ckpt_var = reader.get_variable_to_shape_map()

                    var_to_restore = []
                    for v in tf.global_variables():
                        dict_name = v.name.split(':')[0]
                        if dict_name in ckpt_var:
                            if tuple(ckpt_var[dict_name]) == v.shape:
                                print( self.pre ,'Varibles restored: %s' % v.name)
                                var_to_restore.append(v)
                except Exception as e:
                    print( self.pre ,str(e))
                    var_to_restore = tf.global_variables()

                self.loader = tf.compat.v1.train.Saver(var_to_restore)
            else:
                self.loader = tf.compat.v1.train.Saver(tf.global_variables())
                
            self.saver  = tf.compat.v1.train.Saver(tf.global_variables(),max_to_keep=100)
            
                    
        #############################################################################################
        # SUMMARY
        #############################################################################################     

        with tf.name_scope('summary'):
            # Training state
            tf.compat.v1.summary.scalar("Learning_rate", self.learning_rate)
            #tf.compat.v1.summary.scalar("Global_Step",self.global_step)
            # Losses
            tf.compat.v1.summary.scalar("Classification_loss", self.c_loss)
            tf.compat.v1.summary.scalar("Learning_loss_loss_whole", self.l_loss_w)
            tf.compat.v1.summary.scalar("Learning_loss_loss_split", self.l_loss_s)
            tf.compat.v1.summary.scalar("Total_loss_slit", self.t_loss_s)
            tf.compat.v1.summary.scalar("Total_loss_whole", self.t_loss_w)
            # Metrics
            #tf.compat.v1.summary.scalar("Categorical_Accuracy", self.Categorical_Accuracy)
            #tf.compat.v1.summary.scalar("MAB_learning_loss_whole", self.MAB_whole)
            #tf.compat.v1.summary.scalar("MAB_learning_loss_split", self.MAB_split)
            
            self.merged = tf.summary.merge_all()

            
        #############################################################################################
        # SETUP TENSORFLOE SESSION
        #############################################################################################
        config_tf = tf.ConfigProto(allow_soft_placement=True) 
        config_tf.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config_tf)
        self.sess.graph.as_default()
        self.sess.run(tf.global_variables_initializer())

        #############################################################################################
        # SETUP WATCHER
        #############################################################################################        
        self.run_watcher = get_run_watcher()
        self.run_watcher.add_run.remote(name=self.name_run,
                                        user=self.user,
                                        progress=self.progress,
                                        wandb_url=self.wandb.run.get_url(),
                                        status="Idle")


    
    @ray.method(num_returns = 0)
    def start_training(self):
        import threading
        import os
        import numpy as np
        from copy import deepcopy
        
        def train():
            try:
                import time


                # Make the director to save the trained models
                checkpoint_dir = os.path.join(self.run_dir, 'checkpoint')
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)

                # load an initial weight
                if self.initial_weight_path is not False:
                    try:
                        print( self.pre ,'=> Restoring weights from: %s ... ' % self.initial_weight_path)
                        self.loader.restore(self.sess, self.initial_weight_path)
                    except:
                        print( self.pre ,'=> %s does not exist !!!' % self.initial_weight_path)
                        print( self.pre ,'=> Now it starts to train from scratch ...')
                elif self.transfer_weight_path is not False:
                        print( self.pre ,'=> Restoring weights from: %s ... ' % self.transfer_weight_path)
                        self.loader.restore(self.sess, self.transfer_weight_path)
                else:
                    print( self.pre ,'=> Starts to train from scratch ')


                print( self.pre ,'self.steps_per_epoch '+str(self.steps_per_epoch))
                self.run_watcher.update_run.remote(name=self.name_run, status="Training")

                
                for epoch in (range(self.start_epoch , 1+self.total_epochs )):

                    if self.stop_flag:
                        self.run_watcher.update_run.remote(name=self.name_run, status="Idle")
                        break


                    m_acc_e = np.array([])
                    mae_e   = np.array([])

                    train_epoch_loss, test_epoch_loss = [], []
                    count_nan = 0
                    for step in (range(self.steps_per_epoch)):

                        if self.stop_flag:
                            break

                        if epoch <= self.split_epoch: 
                            result_step= self.sess.run([self.train_op_whole, 
                                                        self.t_loss_w,
                                                        self.merged,
                                                        self.Categorical_Accuracy,
                                                        self.MAE_whole,
                                                        self.global_step])
                        else:
                            result_step= self.sess.run([self.train_op_split, 
                                                        self.t_loss_s,
                                                        self.merged,
                                                        self.Categorical_Accuracy,
                                                        self.MAE_split,
                                                        self.global_step])

                        # get de values to generate the metrics and logs
                        r_t_loss        = result_step[1]
                        summary         = result_step[2]
                        acc_class       = result_step[3]
                        MAE             = result_step[4]
                        global_step_val = result_step[5]
                        
                        # make a vector to estimate the accuracy of the epoch
                        m_acc_e = np.append(m_acc_e,acc_class)
                        mae_e   = np.append(mae_e,MAE)
                        
                        # Save the loss
                        if np.isnan(r_t_loss):
                            count_nan += 1
                        else:
                            train_epoch_loss.append(r_t_loss)

                        #if step % 50 == 0:
                         #   self.wandb.tensorflow.log(summary,step=global_step_val )
                         #   self.wandb.log({"Classification Accuracy Step":m_acc_e.mean(),"Mean Absolute Error Step":mae_e.mean()},step=global_step_val)
                            

                        # update global step
                        self.global_step_val = global_step_val

                        # Update progress
                        if step % 100 == 0:
                            self.progress = round(self.global_step_val / self.global_step_goal * 100.0, 2)
                            self.run_watcher.update_run.remote(name=self.name_run, progress=self.progress)

                    # get the mean epoch total loss
                    train_epoch_loss = np.mean(train_epoch_loss)

                    current_epoch = deepcopy(epoch)

                    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    self.wandb.log({"Train: Classification Accuracy":m_acc_e.mean(),"Train: Mean Absolute Error":mae_e.mean()},step = current_epoch)
                    
                    
                    to_print = ''
                    to_print += "Epoch: %2d / %2d || "%(current_epoch, 1+self.total_epochs)
                    to_print += "Time: %s || "%(log_time)
                    to_print += "Accuracy: %.2f || "%(100*m_acc_e.mean())
                    to_print += "MAE LossNet: %.2f || "%(mae_e.mean())
                    print( self.pre ,to_print)
                    
                    if current_epoch % self.config['RUNS']['test_each'] == 0:
                        
                        ckpt_file = os.path.join(checkpoint_dir, "epoch"+str(current_epoch)+".ckpt")
                        self.saver.save(self.sess, ckpt_file, global_step=current_epoch)
                        """
                        try:
                            detection_test = local_file("test_agent_cifar").Active_Learning_test
                            tester = detection_test.remote( self.config,
                                                            self.test_set,
                                                            self.num_run,
                                                            current_epoch,
                                                            "epoch"+str(current_epoch)+".ckpt-"+str(current_epoch),
                                                            wandb_id=self.test_run_id)

                            if self.test_run_id is None:
                                self.test_run_id = ray.get(tester.get_wandb_id.remote())
                            tester.evaluate.remote()
                            tester.__ray_terminate__.remote()
                        except Exception as e:
                            print( self.pre ,'error')
                            print( self.pre ,e)
                        

                        if self.val_dataset_name is not None and self.val_source is not None:
                            try:
                                detection_val = local_file("val_actor").YoloVal
                                validator = detection_val.remote(self.project,
                                                                self.val_source,
                                                                self.val_dataset_name,
                                                                self.group,
                                                                self.run_dir,
                                                                self.datanet,
                                                                self.dataset_header,
                                                                self.config,
                                                                "epoch"+str(epoch)+".ckpt-"+str(epoch),
                                                                self.global_step_val,
                                                                wandb_id=self.val_run_id, agnostic_eval=self.agnostic_eval)

                                if self.val_run_id is None:
                                    self.val_run_id = ray.get(validator.get_wandb_id.remote())
                                validator.evaluate.remote()
                                validator.__ray_terminate__.remote()
                            except Exception as e:
                                print( self.pre ,e)
                            """
                    self.run_watcher.update_run.remote(name=self.name_run, status="Finished", progress=self.progress)

            except Exception as e:
                self.run_watcher.update_run.remote(name=self.name_run, status="Error")
                print( self.pre ,e)
            
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
        return {"global_step" : self.global_step_val, "progress": self.progress, }