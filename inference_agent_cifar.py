import ray

@ray.remote(num_gpus=1, resources={"gpu_lvl_1" : 1})
class Active_Learning_inference:
    def __init__(self,  config, 
                        un_labeled_set, 
                        num_run,
                        weight_file):
        
        
        #############################################################################################
        # LIBRARIES
        ############################################################################################# 
        import os
        self.run_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.run_path)
        
        from AutoML import AutoMLDataset, local_module
        import tensorflow as tf
        
        core = local_module("core")

        backbones = local_module("backbones")

        from tensorflow.python import pywrap_tensorflow
        import numpy as np
        import pandas as pd
        
        #############################################################################################
        # PARAMETERS 
        #############################################################################################
        self.config         = config
        self.source         = config["PROJECT"]["source"]
        self.run_dir        = os.path.join(config["PROJECT"]["group_dir"],"Stage_"+str(num_run))
        
        self.num_run        = num_run
        self.name_run       = "AL_Inference_"+str(num_run)
        
        self.evaluation_folder = os.path.join(self.run_dir, 'evaluation')
        self.evaluation_file   = os.path.join(self.evaluation_folder, "accuracy_epoch.csv")
        self.ordered_indexes   = os.path.join(self.evaluation_folder, "ordered_indexes.csv")
        
        self.pre ='\x1b[6;30;42m' + self.name_run + '\x1b[0m' #"____" #
        
        # read the evaluation files 
        self.df = pd.read_csv(self.evaluation_file ,index_col=0)
        which_to_use = self.df['accuracy'].idxmax()
        best_epoch   = self.df.loc[which_to_use]['epoch']
        weight_file_v2  = "epoch"+str(best_epoch)+".ckpt-"+str(best_epoch)
        
        self.weight_file       = weight_file #os.path.join(self.run_dir, 'checkpoint', weight_file)

        
        #############################################################################################
        # SETUP WANDB
        #############################################################################################
        import wandb
        self.wandb = wandb

        self.wandb.init(project  = config["PROJECT"]["project"], 
                        group    = config["PROJECT"]["group"], 
                        name     = self.name_run,
                        job_type = "Inference",
                        config   =  config)


        #############################################################################################
        # LOAD DATA
        #############################################################################################
        if self.source=='CIFAR':
            from data_utils import CIFAR10Data
            # Load data
            cifar10_data = CIFAR10Data()
            x_train, y_train, _, _ = cifar10_data.get_data(subtract_mean=False)

            x_infer = x_train[un_labeled_set]
            #y_infer = y_train[un_labeled_set]

        else:
            pass
        
        #############################################################################################
        # DATA GENERATOR
        #############################################################################################
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow(x_infer,
                                       un_labeled_set,
                                       batch_size= self.config["TEST"]["batch_size"],
                                       shuffle=False)
        
        wh = self.config["NETWORK"]["INPUT_SIZE"]
        num_cls = len(self.config["NETWORK"]["CLASSES"])
        
        features_shape = [None, wh, wh, 3]
        labels_shape = [None]
        
        tf_data = tf.data.Dataset.from_generator(lambda: train_gen, 
                                                 output_types=(tf.float32, tf.int64),
                                                output_shapes = (tf.TensorShape(features_shape), tf.TensorShape(labels_shape)))
        data_tensors = tf_data.make_one_shot_iterator().get_next()
        self.img_input = data_tensors[0]
        self.indexes   = data_tensors[1]
        
        
        #############################################################################################
        # GENERATE MODEL
        #############################################################################################
        
        # Get the selected backbone
        self.backbone = getattr(backbones,self.config["PROJECT"]["Backbone"])
        
        with tf.name_scope("define_loss"):           
            # get the classifier
            self.model = core.Classifier_AL(self.backbone, self.config["NETWORK"], reduction='mean')
            
            self.c_pred, self.l_pred_w, self.l_pred_s = self.model.build_nework(self.img_input)
            
            self.embedding = self.model.emb_w
            

        #############################################################################################
        # GLOBAL PROGRESS
        #############################################################################################
        self.evaluation_steps = int(np.ceil(len(x_infer) / self.config['TRAIN']["batch_size"]))

                
        #############################################################################################
        # SETUP TENSORFLOE SESSION
        #############################################################################################
        config_tf = tf.ConfigProto(allow_soft_placement=True) 
        config_tf.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config_tf)

        #############################################################################################
        # LOAD PREVIUS TRAINED MODEL
        #############################################################################################
        #TODO this should load a checkpoint of a training based in the training epoch or
        # this could load any saved model for transfer learning
        self.saver = tf.compat.v1.train.Saver(tf.global_variables())
        print('Restoring from '+str(self.weight_file))
        self.saver.restore(self.sess, self.weight_file)
                    

    @ray.method(num_returns = 1)
    def get_wandb_id(self):
        return self.run_id


    @ray.method(num_returns = 0)
    def evaluate(self):
        
        import os
        import numpy as np
        import pandas as pd
        import time
        import plotly.graph_objects as go
        from sklearn import metrics
        from sklearn import preprocessing
        
        
        #############################################################################################
        # INFER THE TEST SET
        #############################################################################################
        list_np = []
        progress =0

        for step in range(self.evaluation_steps):

            result_step= self.sess.run([self.c_pred,
                                        self.l_pred_w,
                                        self.l_pred_s,
                                        self.embedding,
                                        self.indexes])

            
            if step==0:
                for res in result_step:
                    list_np.append(res)
    
            else:
                for i in range(len(result_step)):
                    list_np[i] = np.concatenate([list_np[i], result_step[i]])
                    
            if step % 100 == 0:
                progress = round(step / self.evaluation_steps * 100.0, 2)
                print(self.pre, "Progress: ", progress, "\%")

        #############################################################################################
        # GET SOME VALUES
        #############################################################################################
        c_pred = list_np[0]
        l_pred_w = list_np[1]
        l_pred_s = list_np[2]
        embedding = list_np[3]
        indexes = list_np[4]
        scores_array = np.max(list_np[0],axis=1)
        
        to_df ={}
        #to_compu['s_uncertanty'] = min_max_scaler.fit_transform(np_list[2].reshape(-1, 1)).squeeze()
        to_df['uncertanty'] = l_pred_w
        to_df['indexes'] = indexes
        
        df_uncertainty =  pd.DataFrame(to_df)
        df_uncertainty.sort_values('uncertanty',ascending=False,inplace=True)
        df_uncertainty.reset_index(drop=True,inplace=True)
        
        df_uncertainty.to_csv(self.ordered_indexes, index=False)
        
        ordered_indexes  =df_uncertainty['indexes'].to_numpy()
        
        
        
