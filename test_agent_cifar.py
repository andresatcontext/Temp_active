import ray

@ray.remote(num_gpus=1)
class Active_Learning_test:
    def __init__(self,  config, 
                        test_set, 
                        num_run,
                        epoch,
                        weight_file, 
                        wandb_id=None):
        
        
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
        
        
        self.group           = "Stage_"+str(num_run)
        self.name_run        = "Test_"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group )
        self.pre ='\x1b[6;30;42m' + self.name_run + '\x1b[0m' #"____" #
        
        self.epoch          = epoch
        
        self.weight_name       = weight_file
        self.weight_file       = os.path.join(self.run_dir, 'checkpoint', weight_file)
        
        self.evaluation_folder = os.path.join(self.run_dir, 'evaluation')
        self.evaluation_file   = os.path.join(self.evaluation_folder, "accuracy_epoch.csv")

        try:
            os.mkdir(self.evaluation_folder)
        except:
            pass
        
        if not os.path.isfile(self.evaluation_file ) or epoch==0:
            self.df = pd.DataFrame(columns=['epoch','accuracy'])
            self.df.to_csv(self.evaluation_file )
        else:
            self.df = pd.read_csv(self.evaluation_file ,index_col=0)
        
        
        #############################################################################################
        # SETUP WANDB
        #############################################################################################
        import wandb
        self.wandb = wandb

        if wandb_id is None:
            self.wandb.init(project  = config["PROJECT"]["project"], 
                            group    = config["PROJECT"]["group"], 
                            name     = "Test_"+str(num_run),
                            job_type = self.group,
                            config   =  config)
            self.run_id = wandb.run.id
        else:
            self.wandb.init(project  = config["PROJECT"]["project"], 
                            group    = config["PROJECT"]["group"], 
                            name     = "Test_"+str(num_run),
                            job_type = self.group,
                            id=wandb_id, 
                            resume=True,
                            config   =  config)
            self.run_id = wandb_id
            

        #############################################################################################
        # LOAD DATA
        #############################################################################################
        if self.source=='CIFAR':
            from data_utils import CIFAR10Data
            # Load data
            cifar10_data = CIFAR10Data()
            _, _, x_test, y_test = cifar10_data.get_data(subtract_mean=False)
            
            #x_test = x_test[test_set]
            #y_test = y_test[test_set]
        else:
            pass
        
        #############################################################################################
        # DATA GENERATOR
        #############################################################################################
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow(x_test,
                                       y_test,
                                       batch_size= self.config["TEST"]["batch_size"],
                                       shuffle=False)
        
        wh = self.config["NETWORK"]["INPUT_SIZE"]
        self.num_class = len(self.config["NETWORK"]["CLASSES"])
        
        features_shape = [None, wh, wh, 3]
        labels_shape = [None, self.num_class]
        
        tf_data = tf.data.Dataset.from_generator(lambda: train_gen, 
                                                 output_types=(tf.float32, tf.float32),
                                                output_shapes = (tf.TensorShape(features_shape), tf.TensorShape(labels_shape)))
        data_tensors = tf_data.make_one_shot_iterator().get_next()
        self.img_input = data_tensors[0]
        self.c_true    = data_tensors[1]
        
        
        #############################################################################################
        # GENERATE MODEL
        #############################################################################################
        self.trainable      = False
        # Get the selected backbone
        self.backbone = getattr(backbones,self.config["PROJECT"]["Backbone"])
        
        with tf.compat.v1.variable_scope("Backbone"):
            c_pred_features = self.backbone(self.img_input, self.num_class, self.trainable)
            self.c_pred = c_pred_features[0]
        
        with tf.compat.v1.variable_scope("LossNet"):
            self.l_pred_w, self.l_pred_s, self.embedding_whole, self.embedding_split = core.Lossnet(c_pred_features, self.config["NETWORK"]["embedding_size"])
            
        with tf.name_scope("Define_loss"):           
            # get the classifier
            self.Losses_compute = core.Loss_Lossnet(margin = self.config["NETWORK"]["MARGIN"])
            self.c_loss_nr, self.c_loss, self.l_loss_w, self.l_loss_s = self.Losses_compute.compute_loss(self.c_true, self.c_pred, self.l_pred_w, self.l_pred_s)
        

        #############################################################################################
        # GLOBAL PROGRESS
        #############################################################################################
        self.evaluation_steps = int(np.ceil(len(x_test) / self.config['TRAIN']["batch_size"]))

           
        #############################################################################################
        # METRICS
        ############################################################################################# 
        with tf.name_scope("define_metrics"):
            
            with tf.name_scope('Categorical_Accuracy'):
                correct_prediction = tf.equal( tf.argmax(self.c_true, 1), tf.argmax(self.c_pred, 1))
                self.Categorical_Accuracy = tf.cast(correct_prediction, tf.float32)
                
            with tf.name_scope('MAE_learning_loss_whole'):
                self.MAE_whole = tf.math.abs(tf.math.subtract(self.c_loss_nr, self.l_loss_w))

            with tf.name_scope('MAE_learning_loss_split'):
                self.MAE_split = tf.math.abs(tf.math.subtract(self.c_loss_nr, self.l_loss_s))
                
                
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
        print(self.pre, 'Restoring from '+str(self.weight_file))
        self.saver.restore(self.sess, self.weight_file)
                    

    @ray.method(num_returns = 1)
    def get_wandb_id(self):
        return self.run_id


    @ray.method(num_returns = 0)
    def evaluate(self):
        
        import os
        import numpy as np
        import time
        import plotly.graph_objects as go
        from sklearn import metrics
        import pandas as pd


        #############################################################################################
        # INFER THE TEST SET
        #############################################################################################
        list_np = []
        for step in range(self.evaluation_steps):

            result_step= self.sess.run([self.c_pred,
                                        self.c_true,
                                        self.c_loss_nr,
                                        self.l_pred_w])
            
            if step==0:
                for res in result_step:
                    list_np.append(res)
    
            else:
                for i in range(len(result_step)):
                    list_np[i] = np.concatenate([list_np[i], result_step[i]])

        #############################################################################################
        # GET VALUES
        #############################################################################################
        pred_array = np.argmax(list_np[0],axis=1)
        annot_array = np.argmax(list_np[1],axis=1)
        scores_array = np.max(list_np[0],axis=1)
        correctness_array = (pred_array==annot_array).astype(np.int64)
        
        true_loss = list_np[2]
        pred_loss = list_np[3]

        
        #############################################################################################
        # COMPUTE METRICS
        #############################################################################################
        
        ######## Classification
        
        # Compute the F1 score, also known as balanced F-score or F-measure
        f1 = metrics.f1_score(annot_array, pred_array, average='macro')
        self.wandb.log({'F1 score': f1}, step=self.epoch)
        
        # Accuracy classification score
        accuracy = metrics.accuracy_score(annot_array, pred_array)
        self.wandb.log({'Test: Classification Accuracy': accuracy}, step=self.epoch)
        
        # Compute Receiver operating characteristic (ROC)
        false_positives_axe, true_positives_axe, roc_seuil = metrics.roc_curve(correctness_array, scores_array)
        fig_roc = self.Plot_ROC(false_positives_axe, true_positives_axe, roc_seuil)
        self.wandb.log({'Receiver operating characteristic (ROC)': fig_roc}, step=self.epoch)
        
        #  Area Under the Curve (AUC) 
        res_auc = metrics.auc(false_positives_axe, true_positives_axe)
        self.wandb.log({'Area Under the Curve (AUC) ': res_auc}, step=self.epoch)  
        
        # Compute confusion matrix to evaluate the accuracy of a classification.
        if len(self.config["NETWORK"]["CLASSES"])<200:
            cm = metrics.confusion_matrix(annot_array, pred_array).astype(np.float32)
            fig_cm = self.Plot_confusion_matrix(cm)
            self.wandb.log({'Confusion Matrix' : fig_cm}, step=self.epoch)
                   
        ######## Regression (loss estimation)
        
        mae_v1 = metrics.mean_absolute_error(true_loss, pred_loss)
        self.wandb.log({'Mean absolute error': mae_v1}, step=self.epoch)

        
        evs_v1 = metrics.explained_variance_score(true_loss, pred_loss)
        self.wandb.log({'Explained variance': evs_v1}, step=self.epoch)
                
        mse_v1 = metrics.mean_squared_error(true_loss, pred_loss)
        self.wandb.log({'Mean squared error': mse_v1}, step=self.epoch)

        #############################################################################################
        # Save accuracy and epoch to select best model
        #############################################################################################
        temp_df= pd.DataFrame({'epoch':[self.epoch],'accuracy':[accuracy]})
        
        pd.concat([self.df,temp_df],ignore_index=True).to_csv(self.evaluation_file)
        
        to_print = 'Test || '
        to_print += "Epoch: %2d || "%(self.epoch)
        to_print += "Accuracy: %.2f || "%(100*accuracy)
        print(self.pre, to_print)

        

                  
    def Plot_confusion_matrix(self,cm):
        
        import plotly.graph_objects as go
        import numpy as np
        
        
        for i in range(cm.shape[0]):
            line_sum = np.sum(cm[i])
            if line_sum != 0:
                for ii in range(cm.shape[1]):
                    cm[i, ii] = float(cm[i, ii])/float(line_sum)

        # Compute and save confusion matrix
        fig = go.Figure({'data': [
                            {
                                'x': self.config["NETWORK"]["CLASSES"],
                                'y': self.config["NETWORK"]["CLASSES"],
                                'z': cm.tolist(),
                                'type': 'heatmap', 'name': 'Confusion_matrix',
                                'colorscale': [[0, 'rgb(255, 255, 255)'],
                                               [0.001, 'rgb(255, 255, 161)'],
                                               [0.25, 'rgb(255, 199, 0)'],
                                               [0.6, 'rgb(123, 189, 255)'],
                                               [1.0, 'rgb(0, 0, 255)']
                                               ]
                            }
                    ],
                    'layout': {
                        'title': 'test confusion matrix',
                        'xaxis': {
                            'constrain': 'domain'
                        },
                        'yaxis': {
                            'scaleanchor': 'x'
                        },
                        'autosize': True
                    }
                })
        return fig
                  
    def Plot_ROC(self,false_positives_axe, true_positives_axe, roc_seuil):
        import plotly.graph_objects as go
        
        # Plot Compute Receiver operating characteristic (ROC)
        fpa = []
        tpa = []
        roc_s = []
        for false_pos, true_pos, seuil in zip(false_positives_axe, true_positives_axe, roc_seuil):
            if len(fpa)==0 or abs(false_pos-fpa[-1]) < 0.01 or abs(true_pos-tpa[-1]) < 0.01 or abs(seuil-roc_s[-1]) < 0.01:
                fpa.append(false_pos)
                tpa.append(true_pos)
                roc_s.append(seuil)
        fig = go.Figure({'data': [
                            {
                                'x': fpa,
                                'y': tpa,
                                'text': roc_s,
                                'type': 'scatter',
                                'mode' : 'lines',
                                'hovertemplate' : '<b>True Positives Rate </b>: %{y:.3f}'
                                                '<br><b>False Positives Rate </b>: %{x:.3f}<br>'
                                                '<b>Threshold </b>: %{text:.3f}',
                            }
                        ],
                        'layout': {
                            'yaxis': {
                                'title': "True Positives Rate",
                                'type': 'linear',
                                'autorange': False,
                                'range': [0.0, 1.0]
                            },
                            'xaxis': {
                                'title': "False Positives Rate",
                                'type': 'linear',
                                'autorange': False,
                                'range': [0.0, 1.0]
                            },
                        }
                        })
        return fig
