from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
import ray

@ray.remote(num_gpus=1)
class Active_Learning_inference:
    def __init__(self,   config, 
                         filenames,
                         labels,
                         classes_semantics,
                         num_run,
                         model_path):

        
        #############################################################################################
        # LIBRARIES
        #############################################################################################        
        import os
        import numpy as np
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.keras import optimizers, losses, models, backend, layers, metrics
        
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
        self.list_classes    = classes_semantics
        self.filenames       = filenames
        self.labels          = labels
        
        self.config          = config
        self.filenames       = filenames
        self.labels          = labels
        self.num_run         = num_run
        self.group           = "Stage_"+str(num_run)
        self.name_run        = "Test"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group)
        self.run_dir_check   = os.path.join(self.run_dir ,'checkpoints')
        self.checkpoints_path= os.path.join(self.run_dir_check,'checkpoint.{epoch:03d}.hdf5')
        self.user            = get_user()
        
        self.transfer_weight_path = self.config['TRAIN']["transfer_weight_path"]
        self.input_shape       = [self.config["NETWORK"]["INPUT_SIZE"], self.config["NETWORK"]["INPUT_SIZE"], 3]
        
        self.pre ='\033[1;36m' + self.name_run + '\033[0;0m' #"____" #
        self.problem ='\033[1;31m' + self.name_run + '\033[0;0m'
        
        # Creating the test folder
        self.ordered_indexes   = os.path.join(self.run_dir, "ordered_indexes.csv")
        
        try:
            os.mkdir(self.evaluation_folder)
        except:
            pass
        
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
                # LOAD DATA
                #############################################################################################
                self.DataGen = data_pipeline.ClassificationDataset_AL(  config["TEST"]["batch_size"],
                                                                            self.filenames,
                                                                            self.labels,
                                                                            self.list_classes,
                                                                            subset = "inference",
                                                                            original_size      = config["DATASET"]["original_size"],
                                                                            data_augmentation  = False)  
                self.num_class = len(self.DataGen.list_classes)
                
                #############################################################################################
                # GLOBAL PROGRESS
                #############################################################################################
                self.steps_per_epoch  = int(np.ceil(self.DataGen.nb_elements/config["TEST"]["batch_size"]))
                print(self.pre,'Number of elements in the inference set', self.DataGen.nb_elements)
                
                
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
                #c_pred_1 = c_pred_features_1[0]
                loss_pred_embeddings = lossnet.Lossnet(c_pred_features, self.config["NETWORK"]["embedding_size"])
                
                # add some inputs to prediction and testing
                labels_tensor = tf.keras.Input(tensor=self.DataGen.next_element[1], name= 'labels_tensor')
                files_tesor   = tf.keras.Input(tensor=self.DataGen.next_element[2], name= 'files_tesor')
                
                model_inputs  = [img_input, labels_tensor, files_tesor]
                model_outputs = [c_pred, loss_pred_embeddings[0], loss_pred_embeddings[2], labels_tensor, files_tesor]
                
                self.model = models.Model(inputs=model_inputs, outputs=model_outputs)


                #############################################################################################
                # LOAD PATH TO INFER
                #############################################################################################

                self.loaded_epoch = int(model_path.split('.')[-2])
                print(self.pre, "Loading weigths from: ",model_path)
                print(self.pre, "The detected epoch is: ",self.loaded_epoch)
                # load weigths
                self.model.load_weights(model_path)

                #############################################################################################
                # INIT VARIABLES
                #############################################################################################
                #self.sess.graph.as_default()
                backend.set_session(self.sess)
                self.sess.run(tf.local_variables_initializer())

                ################################################################
                # SETUP WATCHER
                ################################################################    
                print(self.pre,'Init done')

    @ray.method(num_returns = 0)
    def inference(self):
        import numpy as np
        from sklearn import metrics
        import pandas as pd
        
        with self.graph.as_default():
            with self.sess.as_default():
                
                # set the dataset to the beggining
                self.sess.run(self.DataGen.iterator.initializer)
                
                print( self.pre ,"Start inference")
                results = self.model.predict(None,steps=self.steps_per_epoch)
                
                for i, res in enumerate(results):
                    results[i] = res[:self.DataGen.nb_elements]
                    print(results[i].shape)
                
                
                #############################################################################################
                # GET VALUES
                #############################################################################################
                # c_pred_1, loss_pred_embeddings[0], loss_pred_embeddings[2], labels_tensor, files_tesor]
                pred_array   = np.argmax(results[0],axis=1)
                annot_array  = np.squeeze(results[3])
                scores_array = np.max(results[0],axis=1)
                pred_loss    = results[1][:,-1]
                files_names  = results[4]
         

                # Create dataframe and save it

                to_df ={}
                #to_compu['s_uncertanty'] = min_max_scaler.fit_transform(np_list[2].reshape(-1, 1)).squeeze()
                to_df['uncertanty'] = pred_loss
                to_df['files']      = files_names
                to_df['scores']     = scores_array
                to_df['prediction'] = pred_array
                to_df['labels']     = annot_array
                

                df_uncertainty =  pd.DataFrame(to_df)
                df_uncertainty.sort_values('uncertanty',ascending=False,inplace=True)
                df_uncertainty.reset_index(drop=True,inplace=True)
                df_uncertainty.to_csv(self.ordered_indexes, index=False)

                #ordered_indexes  = df_uncertainty['indexes'].to_numpy()
        