#RUN_AT_FATHER = False

#from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
#import ray

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
'''
from tqdm import tqdm

class AL_train_cifar():

    def __init__(self,config, labeled_set, project="AL_CIFAR" , source=None, dataset_name=None, val_source=None, val_dataset_name=None, bs=32, agnostic_eval=False):
        
        import os
        from core.Classifier_AL import Classifier_AL
        from backbones.resnet18_paper import ResNet18
        from tensorflow.python import pywrap_tensorflow
        import numpy as np
        
        self.backbone = ResNet18
        
        
        #############################################################################################
        # PARAMETERS RUN
        #############################################################################################
        self.project = project
        self.runs_folder = "./runs"
        self.run_dir, self.name_run = self.get_run_dir()
        self.group                  = "TEST_AL"
        self.user                   = "Andres"
        
        self.initial_weight =False
        
        # Creating the Run folder
        import shutil
        if os.path.exists(self.run_dir) and self.initial_weight is False:
            shutil.rmtree(self.run_dir)
        if os.path.exists(self.run_dir) is False:
            os.mkdir(self.run_dir)
        
        
        #############################################################################################
        # SETUP TENSORFLOE SESSION
        #############################################################################################
        config_tf = tf.ConfigProto(allow_soft_placement=True) 
        config_tf.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config_tf)
        
        
        #############################################################################################
        # LOAD DATA
        #############################################################################################
        from data_utils import CIFAR10Data
        # Load data
        cifar10_data = CIFAR10Data()
        num_classes = len(cifar10_data.classes)
        x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=False)

        x_train = x_train[labeled_set]
        y_train = y_train[labeled_set]
        
        
        #############################################################################################
        # SETUP WANDB
        #############################################################################################
        import wandb
        self.wandb = wandb
        self.wandb.init(project=self.project, group=self.group, job_type="train", sync_tensorboard=True,config=config)
        self.config = self.wandb.config
        
        
        #############################################################################################
        # LOAD DATA
        #############################################################################################
        train_datagen = ImageDataGenerator(
                    width_shift_range=self.config.width_shift_range,
                    height_shift_range=self.config.height_shift_range,
                    horizontal_flip=self.config.horizontal_flip)

        train_gen = train_datagen.flow(x_train,
                                       y_train,
                                       batch_size=self.config.batch_size)
        
        features_shape = [None, 32, 32, 3]
        labels_shape = [None, 10]
        

        
        tf_data = tf.data.Dataset.from_generator(lambda: train_gen, 
                                                 output_types=(tf.float32, tf.float32),
                                                output_shapes = (tf.TensorShape(features_shape), tf.TensorShape(labels_shape)))
        #tf_data = tf_data.apply(tf.contrib.data.ignore_errors())
        data_tensors = tf_data.make_one_shot_iterator().get_next()
        
        tf_data = tf_data.repeat()
        #tf_data = tf_data.batch(self.config.batch_size)
        #tf_data = tf_data.prefetch(100)
        #data_tensors = tf_data.make_initializable_iterator().get_next()
        self.img_input = data_tensors[0]
        self.c_true    = data_tensors[1]
        
        
        
        #############################################################################################
        # GENERATE MODEL
        #############################################################################################
        with tf.name_scope("define_loss"):
            # define inputs to test 
            #self.img_input = layers.Input(shape=self.config.input_shape, name="img_input")
            #c_true = layers.Input(shape=(self.config.num_class), name="c_true")
            
            # get the classifier
            self.model = Classifier_AL(self.backbone, self.config)
            self.c_pred, self.l_pred_w, self.l_pred_s = self.model.build_nework(self.img_input)
            # get global variables
            self.net_var = tf.global_variables()

            self.c_loss, self.l_loss_w, self.l_loss_s, self.l_true = self.model.compute_loss(self.c_true)
            
            self.t_loss_w =  (self.config.w_c_loss*self.c_loss) + (self.config.w_l_loss * self.l_loss_w)
            self.t_loss_s =  (self.config.w_c_loss*self.c_loss) + (self.config.w_l_loss * self.l_loss_s)
        

        
            
        #############################################################################################
        # LOAD PREVIUS TRAINED MODEL
        #############################################################################################
        #TODO this should load a checkpoint of a training based in the training epoch or
        # this could load any saved model for transfer learning
        self.base_weight_path = False
        with tf.name_scope('loader_and_saver'):
            if self.base_weight_path is not False and self.start_epoch == 0:
                try:
                    reader = pywrap_tensorflow.NewCheckpointReader(self.base_weight_path)
                    ckpt_var = reader.get_variable_to_shape_map()

                    var_to_restore = []
                    for v in self.net_var:
                        dict_name = v.name.split(':')[0]
                        if dict_name in ckpt_var:
                            if tuple(ckpt_var[dict_name]) == v.shape:
                                print('Varibles restored: %s' % v.name)
                                var_to_restore.append(v)
                except Exception as e:
                    print(str(e))
                    var_to_restore = self.net_var

                self.loader = tf.compat.v1.train.Saver(var_to_restore)
            else:
                self.loader = tf.compat.v1.train.Saver(self.net_var)
            self.saver  = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=40)
        
            
        #############################################################################################
        # GLOBAL PROGRESS
        #############################################################################################
        self.start_epoch  = self.config.start_epoch
        self.total_epochs = self.config.EPOCH_WHOLE +self.config.EPOCH_SLIT 
        self.steps_per_epoch = int(np.ceil(self.config.NUM_TRAIN / self.config.batch_size))
        self.global_step_val = self.steps_per_epoch * self.start_epoch
        self.global_step_goal = self.steps_per_epoch * self.total_epochs
        self.progress = round(self.global_step_val / self.global_step_goal * 100.0, 2)
        
        
        #############################################################################################
        # DEFINE LEARNING RATE FOR TRAINING
        #############################################################################################
        with tf.name_scope('learning_rate'):
            # define the current step as tf variable
            self.global_step = tf.Variable(float(self.global_step_val), dtype=tf.float64, trainable=False, name='global_step')
            
            # define warmup_steps in tf
            warmup_steps = tf.constant(self.config.EPOCH_WARMUP * self.steps_per_epoch, dtype=tf.float64, name='warmup_steps')
            # define train_steps in tf
            train_steps = tf.constant((self.config.EPOCH_WHOLE +self.config.EPOCH_SLIT )*self.steps_per_epoch, dtype=tf.float64, name='train_steps')
            
            steps_per_epoch = tf.constant(self.steps_per_epoch, dtype=tf.float64, name='steps_per_epoch')
            MILESTONES = tf.constant(self.config.MILESTONES, dtype=tf.float64)
            
            self.learning_rate = tf.cond(   pred = self.global_step < warmup_steps,
                                            true_fn=lambda: self.global_step / warmup_steps * self.config.lr,
                                            false_fn=lambda: self.config.lr ** (self.config.gamma ** (tf.compat.v1.reduce_sum(tf.cast(MILESTONES<(self.global_step/steps_per_epoch), tf.float64)))))
            
            global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)
            
            
        #############################################################################################
        # DEFINE LEARNING RATE FOR TRAINING
        #############################################################################################
        #with tf.name_scope("define_weight_decay"):
        #    moving_ave = tf.train.ExponentialMovingAverage(self.config.wdecay).apply(tf.trainable_variables())
            
            
        #############################################################################################
        # DEFINE THE PARAMETERS TO TRAIN
        #############################################################################################
        
        with tf.name_scope("define_train_whole"):
            optimizer_train_whole = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss_w, var_list=tf.trainable_variables())
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([optimizer_train_whole, global_step_update]):
                    #with tf.control_dependencies([moving_ave]):
                    self.train_op_whole = tf.no_op()

        with tf.name_scope("define_train_split"):
            optimizer_train_split = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.t_loss_s, var_list=tf.trainable_variables())
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([optimizer_train_split, global_step_update]):
                    #with tf.control_dependencies([moving_ave]):
                    self.train_op_spit = tf.no_op()
                    
        
        
        #############################################################################################
        # METRICS
        ############################################################################################# 
        
        with tf.name_scope("define_metrics"):
            
            with tf.name_scope('Categorical_Accuracy'):
                correct_prediction = tf.equal( tf.argmax(self.c_true, 1), tf.argmax(self.c_pred, 1))
                self.Categorical_Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
            with tf.name_scope('MAB_learning_loss_whole'):
                self.MAB_whole = tf.reduce_mean(tf.math.abs(tf.math.subtract(self.l_true, self.l_loss_w)))

            with tf.name_scope('MAB_learning_loss_split'):
                self.MAB_split = tf.reduce_mean(tf.math.abs(tf.math.subtract(self.l_true, self.l_loss_s)))

        '''
        self.metrics = {}
        # Metrics and trackers
        # Accuraxy classification
        self.metrics["Accuracy Classification"] = metrics.CategoricalAccuracy( name='Accuracy Classification')
        # Mean average error for the lossnet loss
        # this just will compute the mean then we need to add metrics.MAE *because of the tensorflow version
        self.metrics["MAE LossNet loss"]    = metrics.Mean(name="MAE LossNet loss")
        # Mean for the losses
        self.metrics["Classification loss"] = metrics.Mean(name="Classification loss")
        self.metrics["LossNet loss"]        = metrics.Mean(name="LossNet loss")
        self.metrics["Total loss"]          = metrics.Mean(name="Total loss")
                
        
        self.mae_w = metrics.mae(self.l_true, self.l_loss_w)
        self.mae_s =metrics.mae(self.l_true, self.l_loss_s)
        
        
        self.metrics["Accuracy Classification"].update_state(r_c_true, r_c_pred)
        self.metrics["MAE LossNet loss"].update_state(r_mae)
        self.metrics["Classification loss"].update_state(r_c_loss)
        self.metrics["LossNet loss"].update_state(r_l_loss)
        self.metrics["Total loss"].update_state(r_t_loss)
          '''      
                    
        #############################################################################################
        # SUMMARY
        #############################################################################################     

        with tf.name_scope('summary'):
            # Training state
            tf.compat.v1.summary.scalar("Learning_rate", self.learning_rate)
            tf.compat.v1.summary.scalar("Global_Step",self.global_step)
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

        

        
        self.sess.graph.as_default()
        self.sess.run(tf.global_variables_initializer())

        #self.run_watcher = get_run_watcher()
        #self.run_watcher.add_run.remote(name=self.name_run,
        #                                user=self.user,
        #                                progress=self.progress,
        #                                wandb_url=self.wandb.run.get_url(),
        #                                status="Idle")

                
    def get_run_dir(self):
        import glob
        import os

        model_name = self.project.lower().replace(' ', '_')
        runs = glob.glob(os.path.join(self.runs_folder, model_name + '_AL_*'))
        run_id = len(runs)
        return os.path.join(self.runs_folder, model_name + '_AL_' + str(run_id)),  model_name + '_AL_' + str(run_id)
    
    
    def train(self):
        self.stop_flag=False
        import os
        
    #try:
        import time

        
        # Make the director to save the trained models
        checkpoint_dir = os.path.join(self.run_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        # load an initial weight
        if self.initial_weight is not False:
            try:
                print('=> Restoring weights from: %s ... ' % os.path.join(checkpoint_dir, self.initial_weight))
                self.loader.restore(self.sess, os.path.join(checkpoint_dir, self.initial_weight))
            except:
                print('=> %s does not exist !!!' % self.initial_weight)
                print('=> Now it starts to train from scratch ...')
                self.first_stage_epochs = 0
        elif self.base_weight_path is not False:
                print('=> Restoring weights from: %s ... ' % self.base_weight_path)
                self.loader.restore(self.sess, self.base_weight_path)
                if self.start_epoch > 0:
                    self.first_stage_epochs = 0
        else:
            print('=> Starts to train from scratch ')

        
        print('self.steps_per_epoch '+str(self.steps_per_epoch))
        #self.run_watcher.update_run.remote(name=self.name_run, status="Training")
        for epoch in (range(self.start_epoch + 1, 1+(self.config.EPOCH_WHOLE +self.config.EPOCH_SLIT ))):
            if self.stop_flag:
                #self.run_watcher.update_run.remote(name=self.name_run, status="Idle")
                break
                
            # define witch graph to optimize
            if epoch <= self.config.EPOCH_WHOLE:
                train_op = self.train_op_whole
            else:
                train_op = self.train_op_split
                
            metrics_mean = {}


            train_epoch_loss, test_epoch_loss = [], []
            count_nan = 0
            for step in (range(self.steps_per_epoch)):
                
                if self.stop_flag:
                    break
                
                """
                # runs the session
                if epoch <= self.config.EPOCH_WHOLE:
                    # whole model
                    _, Classification_loss, Learning_loss_loss, l_true, loss, global_step_val = self.sess.run(
                        [train_op, self.Classification_loss, self.Learning_loss_loss_whole, self.l_true, self.loss_whole, self.global_step], feed_dict={self.trainable: True})
                else:
                    # split model
                    _, Classification_loss, Learning_loss_loss, l_true, loss, global_step_val = self.sess.run(
                        [train_op, self.Classification_loss, self.Learning_loss_loss_split, self.l_true, self.loss_split, self.global_step], feed_dict={self.trainable: True})
                """
                
                if epoch <= self.config.EPOCH_WHOLE: 
                    result_step= self.sess.run([train_op, 
                                                self.t_loss_w,
                                                self.merged,
                                                self.Categorical_Accuracy,
                                                self.MAB_whole,
                                                self.global_step])
                else:
                    result_step= self.sess.run([train_op, 
                                                self.t_loss_s,
                                                self.merged,
                                                self.Categorical_Accuracy,
                                                self.MAB_split,
                                                self.global_step])
                    

                    
                
                # get de values to generate the metrics and logsh

                r_t_loss        = result_step[1]
                summary         = result_step[2]
                acc_class       = result_step[3]
                MAB             = result_step[4]
                global_step_val = result_step[5]


                self.wandb.tensorflow.log(summary)
                self.wandb.log({"acc_class":acc_class,"MAB":MAB})
                
                # Save the loss
                if np.isnan(r_t_loss):
                    count_nan += 1
                else:
                    train_epoch_loss.append(r_t_loss)
                    
                if step % 50 == 0:
                    print(10*"=",global_step_val)
                    #self.wandb.tensorflow.log(summary)
                    #self.wandb.log({"acc_class":acc_class,"MAB":MAB})
                    #self.wandb.log(logs,step=global_step_val)
                    

                # update global step
                self.global_step_val = global_step_val

                # Update progress
                if step % 100 == 0:
                    self.progress = round(self.global_step_val / self.global_step_goal * 100.0, 2)
                    #self.run_watcher.update_run.remote(name=self.name_run, progress=self.progress)

            # get the mean epoch total loss
            train_epoch_loss = np.mean(train_epoch_loss)

            
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("Epoch: %2d Time: %s Train loss: %.2f Count NaN: %d Saving \n"%(epoch, log_time, train_epoch_loss, count_nan))
            ckpt_file = os.path.join(checkpoint_dir, "epoch"+str(epoch)+".ckpt")
            self.saver.save(self.sess, ckpt_file, global_step=epoch)
            
            
            """
            #try:

             #   detection_test = local_file("test_actor").YoloTest
             #   tester = detection_test.remote(self.project,
             #                                   self.source,
             #                                   self.dataset_name,
             #                                   self.group,
             #                                   self.run_dir,
             #                                   self.datanet,
             #                                   self.dataset_header,
             #                                   self.config,
             #                                   "epoch"+str(epoch)+".ckpt-"+str(epoch),
             #                                   self.global_step_val,
             #                                   wandb_id=self.test_run_id, agnostic_eval=self.agnostic_eval)

              #  if self.test_run_id is None:
              #      self.test_run_id = ray.get(tester.get_wandb_id.remote())
              #  tester.evaluate.remote()
              #  tester.__ray_terminate__.remote()
            #except Exception as e:
            #    print(e)

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
                    print(e)
    #    self.run_watcher.update_run.remote(name=self.name_run, status="Finished", progress=self.progress)
    #except Exception as e:
    #    self.run_watcher.update_run.remote(name=self.name_run, status="Error")
    #    print(e)
    """
if __name__ == "__main__":
    # execute only if run as a script
    import random

    #############################################################################################
    # LOAD DATA
    #############################################################################################
    from data_utils import CIFAR10Data
    # Load data
    cifar10_data = CIFAR10Data()
    num_classes = len(cifar10_data.classes)
    x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)

    #############################################################################################
    # LOAD CONFIG
    #############################################################################################
    from config import Load_config
    config = Load_config(len(x_train),x_train.shape[1:],num_classes,active_learning = False)

    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
    indices = list(range(config["NUM_TRAIN"]))
    random.shuffle(indices)
    labeled_set = indices[:config["ADDENDUM"]]
    unlabeled_set = indices[config["ADDENDUM"]:]
    
    AL_train_cifar_w =  AL_train_cifar(config, labeled_set)
    
    AL_train_cifar_w.train()