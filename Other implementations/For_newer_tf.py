#RUN_AT_FATHER = False

#from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
#import ray

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np



from tqdm import tqdm

class AL_train_cifar():

    def __init__(self, config, labeled_set, project="AL_CIFAR" , source=None, dataset_name=None, val_source=None, val_dataset_name=None, bs=32, agnostic_eval=False):
        
        import os

        from backbones.resnet18_paper import ResNet18
        from core.Classifier_AL import Classifier_AL, Lossnet,Loss_Lossnet
        from tensorflow.keras import layers, Model, losses, metrics, regularizers
        
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
        # LOAD DATA
        #############################################################################################
        from data_utils import CIFAR10Data
        # Load data
        cifar10_data = CIFAR10Data()
        num_classes = len(cifar10_data.classes)
        x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)
        
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
        # DATA GENERATOR
        #############################################################################################
        train_datagen = ImageDataGenerator(
                    width_shift_range=self.config.width_shift_range,
                    height_shift_range=self.config.height_shift_range,
                    horizontal_flip=self.config.horizontal_flip)

        self.train_gen = train_datagen.flow(x_train,
                                       y_train,
                                       batch_size=self.config.batch_size)
        

        #############################################################################################
        # DEFINE CLASSIFIER MODEL
        #############################################################################################
        # create keras model of the classifier
        img_input = layers.Input(shape=self.config.input_shape, name="img_input")
        # using the Backbone
        pred_class_features = self.backbone(img_input,self.config.num_class)
        # Generate model classifier
        self.Classifier = Model(inputs=img_input, outputs=pred_class_features, name="Classifier")
        
        
        #############################################################################################
        # DEFINE LOSSNET MODEL
        #############################################################################################
        # create inputs fot the loss net model
        input_loss_net =[]
        # generate inputs to loss net
        for i, feat_class in enumerate(pred_class_features):
            if i>0:
                input_loss_net.append(layers.Input(shape=feat_class.shape[1:], name="pred_feat_"+str(i-1)))

        # Get class for lossnet
        loss_net = Lossnet(self.config)
        # generate keras model for lossnet
        pred_loss = loss_net.build_nework(input_loss_net)
        self.Loss_net = Model(inputs=input_loss_net, outputs=pred_loss, name="Lossnet")
        
        
        #############################################################################################
        # APPLY REGULARIZATION TO THE MODELS
        #############################################################################################
        # Implement weight_decay
        if self.config.wdecay is not None:
            for layer in self.Classifier.layers+self.Loss_net.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer= regularizers.l2(self.config.wdecay)
        
        
        #############################################################################################
        # DEFINE LOSSES
        #############################################################################################
        self.Loss_lerning_Class  = Loss_Lossnet(self.config)
        #self.Loss_lerning_fn = Loss_Lossnet.run
        self.Classification_loss_fn = losses.CategoricalCrossentropy(name='Classification_Loss')
        
        #############################################################################################
        # FLAG 
        #############################################################################################
        self.train_individual =  False # Flag to train both the models or train each model individually
        
        #############################################################################################
        # DEFINE OPTIMIZER
        #############################################################################################
        self.optimizer = tf.keras.optimizers.SGD( lr=self.config.lr, momentum=self.config.momentum)
        
        
        #############################################################################################
        # DEFINE METRICS
        ############################################################################################  
        self.metrics = {}
        # Metrics and trackers
        # Accuraxy classification
        self.metrics["Accuracy Classification"] = metrics.CategoricalAccuracy( name='Accuracy Classification')
        # Mean average error for the lossnet loss
        # this just will compute the mean then we need to add metrics.MAE *because of the tensorflow version
        self.metrics["MAE Learning loss loss"]  = metrics.Mean(name="MAE Learning loss loss")
        # Mean for the losses
        self.metrics["Classification loss"] = metrics.Mean(name="Classification loss")
        self.metrics["Learning loss loss"]  = metrics.Mean(name="Learning loss loss")
        self.metrics["Total loss"]          = metrics.Mean(name="Total loss")
        
        
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
        '''
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
         '''
        #############################################################################################
        # LOAD PREVIUS TRAINED MODEL
        #############################################################################################
        # this could load any saved model for transfer learning
        self.base_weight_path = False


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
                    _, Total_loss, summary, global_step_val= self.sess.run([train_op, 
                                                                            self.Classification_loss,
                                                                            self.merged, 
                                                                            self.global_step], 
                                                                           feed_dict={self.trainable: True})
                else:
                    _, Total_loss, summary, global_step_val= self.sess.run([train_op, 
                                                                            self.Classification_loss,
                                                                            self.merged, 
                                                                            self.global_step], 
                                                                           feed_dict={self.trainable: True})
                # Save the loss
                if np.isnan(Total_loss):
                    count_nan += 1
                else:
                    train_epoch_loss.append(Total_loss)
                    
                self.wandb.tensorflow.log(summary,step=global_step_val)
                if step % 50 == 0:
                    print(10*"=",global_step_val)
                    #self.wandb.tensorflow.log(summary,step=global_step_val)
                    #print(summary)
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