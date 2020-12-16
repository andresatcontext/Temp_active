import ray

@ray.remote(num_gpus=1)
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
        from tensorflow.keras import backend, layers, models, utils
        
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
        
        #self.evaluation_folder = os.path.join(self.run_dir, 'evaluation')
        #self.evaluation_file   = os.path.join(self.evaluation_folder, "accuracy_epoch.csv")
        self.ordered_indexes   = os.path.join(self.run_dir, "ordered_indexes.csv")
        
        self.pre ='\x1b[6;30;42m' + self.name_run + '\x1b[0m' #"____" #
        
        self.num_class       = len(self.config["NETWORK"]["CLASSES"])
        self.input_shape     = [self.config["NETWORK"]["INPUT_SIZE"], self.config["NETWORK"]["INPUT_SIZE"], 3]
        
        # read the evaluation files 
        #self.df = pd.read_csv(self.evaluation_file ,index_col=0)
        #which_to_use = self.df['accuracy'].idxmax()
        #best_epoch   = self.df.loc[which_to_use]['epoch']
        #weight_file_v2  = "epoch"+str(best_epoch)+".ckpt-"+str(best_epoch)
        
        self.weight_file       = weight_file #os.path.join(self.run_dir, 'checkpoint', weight_file)

        
        #############################################################################################
        # SETUP WANDB
        #############################################################################################
        #import wandb
        #self.wandb = wandb

        #self.wandb.init(project  = config["PROJECT"]["project"], 
        #                group    = config["PROJECT"]["group"], 
        ##                name     = "Inference_"+str(num_run),
         #               job_type = self.group,
         #               config   =  config)
        

        #######################################################################
        # SETUP TENSORFLOW SESSION
        #########################################################################

        self.graph = tf.Graph()
        with self.graph.as_default():
            config_tf = tf.ConfigProto(allow_soft_placement=True) 
            config_tf.gpu_options.allow_growth = True 
            self.sess = tf.Session(config=config_tf,graph=self.graph)
            with self.sess.as_default():
                
                ############################################################
                # LOAD DATA
                ############################################################
                if self.source=='CIFAR':
                    from data_utils import CIFAR10Data
                    # Load data
                    cifar10_data = CIFAR10Data()
                    x_train, y_train, _, _ = cifar10_data.get_data(subtract_mean=False)

                    x_infer = x_train[un_labeled_set]
                    #y_infer = y_train[un_labeled_set]
                else:
                    pass
                    
                ###########################################################################
                # DATA GENERATOR
                ###########################################################################
                self.Data_Generator = core.Generator_cifar_inference(x_infer, un_labeled_set, config)

                ##########################################################################
                # DEFINE CLASSIFIER
                ##########################################################################
                # set input
                img_input = tf.keras.Input(self.input_shape,name= 'input_image')

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

                #################################################################
                # DEFINE FULL MODEL
                #################################################################
                c_pred_features_1 = self.classifier(img_input)
                c_pred_1 = c_pred_features_1[0]

                # define lossnet
                loss_pred_embeddings = core.Lossnet(c_pred_features_1, self.config["NETWORK"]["embedding_size"])

                self.model = models.Model(inputs=img_input, outputs=[c_pred_1]+loss_pred_embeddings)
                
                ################################################################
                # INIT VARIABLES
                ################################################################
                #self.sess.graph.as_default()
                backend.set_session(self.sess)
                self.sess.run(tf.local_variables_initializer())
                
                print(self.pre,'Init done')
                

    @ray.method(num_returns = 0)
    def inference(self):
        
        import os
        import numpy as np
        import pandas as pd
        import time
        import plotly.graph_objects as go
        from sklearn import metrics
        from sklearn import preprocessing
        
        
        with self.graph.as_default():
            with self.sess.as_default():

                print(self.pre, 'Restoring from '+str(self.weight_file))
                self.model.load_weights(self.weight_file)

                #########################################################
                # INFER THE UNLABELED SET
                ###############################################################

                for i, (X,index_x) in enumerate(self.Data_Generator):
                    preds  = self.model.predict(X)
                    if i==0:
                        indexes   = index_x
                        c_pred    = preds[0]
                        l_pred_w  = preds[1][:,-1]
                        l_pred_s  = preds[2][:,-1]
                        embedding = preds[3]
                    else :
                        indexes   = np.concatenate([indexes,index_x])  
                        c_pred    = np.concatenate([c_pred,preds[0]])
                        l_pred_w  = np.concatenate([l_pred_w,preds[1][:,-1]])
                        l_pred_s  = np.concatenate([l_pred_s,preds[2][:,-1]])
                        embedding = np.concatenate([embedding,preds[3]])


        # get the score max
        scores_array = np.max(c_pred,axis=1)
        
        # Create dataframe and save it
        
        to_df ={}
        #to_compu['s_uncertanty'] = min_max_scaler.fit_transform(np_list[2].reshape(-1, 1)).squeeze()
        to_df['uncertanty'] = l_pred_w
        to_df['indexes'] = indexes
        
        df_uncertainty =  pd.DataFrame(to_df)
        df_uncertainty.sort_values('uncertanty',ascending=False,inplace=True)
        df_uncertainty.reset_index(drop=True,inplace=True)
        
        df_uncertainty.to_csv(self.ordered_indexes, index=False)
        
        ordered_indexes  = df_uncertainty['indexes'].to_numpy()
        
        
        
