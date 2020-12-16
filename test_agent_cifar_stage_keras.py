import ray

@ray.remote(num_gpus=1)
class Active_Learning_test_stage:
    def __init__(self,  config , num_run, test_set):
        
        
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
        self.config        = config
        self.source        = config["PROJECT"]["source"]
        self.num_run       = num_run
        self.group         = "Stage_"+str(num_run)
        self.name_run      = "Test_"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group)
        self.run_dir_check   = os.path.join(self.run_dir ,'checkpoints')
        self.checkpoints_path= os.path.join(self.run_dir_check,'checkpoint.{epoch:03d}.hdf5')
        #self.user            = get_user()
        self.stop_flag       = False
        
        self.num_class       = len(self.config["NETWORK"]["CLASSES"])
        self.input_shape     = [self.config["NETWORK"]["INPUT_SIZE"], self.config["NETWORK"]["INPUT_SIZE"], 3]
        
        self.pre ='\033[1;36m' + self.name_run + '\033[0;0m' #"____" #
        self.problem ='\033[1;31m' + self.name_run + '\033[0;0m'
        
        self.evaluation_folder = os.path.join(self.run_dir, 'evaluation')
        self.evaluation_file   = os.path.join(self.evaluation_folder, "accuracy_epoch.csv")
        
        try:
            os.mkdir(self.evaluation_folder)
        except:
            pass

        # list the epochs to evaluate
        files_run_dir_check = os.listdir(self.run_dir_check)
        self.epochs_checked = [int(i.split('.')[-2]) for i in files_run_dir_check if i.endswith('hdf5')]
        self.epochs_checked.sort()

        #######################################################################
        # SETUP TENSORFLOW SESSION
        #########################################################################

        self.graph = tf.Graph()
        with self.graph.as_default():
            config_tf = tf.ConfigProto(allow_soft_placement=True) 
            config_tf.gpu_options.allow_growth = True 
            self.sess = tf.Session(config=config_tf,graph=self.graph)
            with self.sess.as_default():
        
                ###################################################################
                # SETUP WANDB
                ####################################################################
                import wandb

                self.wandb = wandb
                self.wandb.init(project  = config["PROJECT"]["project"], 
                                group    = config["PROJECT"]["group"], 
                                name     = "Test_"+str(num_run),
                                job_type = self.group ,
                                sync_tensorboard = True,
                                config = config)
                ####################################################################
                # LOAD DATA
                #####################################################################
                if self.config["PROJECT"]["source"]=='CIFAR':
                    from data_utils import CIFAR10Data
                    # Load data
                    cifar10_data = CIFAR10Data()
                    _, _, x_test, y_test = cifar10_data.get_data(normalize_data=False)
                else:
                    raise NameError('This is not implemented yet')
                    
                ###########################################################################
                # DATA GENERATOR
                ###########################################################################
                self.Data_Generator = core.Generator_cifar_test(x_test, y_test, config)


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

                ###################################################################
                # DEFINE FULL MODEL
                ####################################################################
                c_pred_features_1 = self.classifier(img_input)
                c_pred_1 = c_pred_features_1[0]

                # define lossnet
                loss_pred_embeddings = core.Lossnet(c_pred_features_1, self.config["NETWORK"]["embedding_size"])

                self.model = models.Model(inputs=img_input, outputs=[c_pred_1]+loss_pred_embeddings)
                
        
                ##################################################################
                # DEFINE CALLBACKS
                ##################################################################
                # Checkpoint saver
                self.callbacks = []

                # Callback to wandb
                self.callbacks.append(self.wandb.keras.WandbCallback())
                
                ################################################################
                # INIT VARIABLES
                ################################################################
                #self.sess.graph.as_default()
                backend.set_session(self.sess)
                self.sess.run(tf.local_variables_initializer())
                
                print(self.pre,'Init done')
                

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

        with self.graph.as_default():
            with self.sess.as_default():
                for epoch in self.epochs_checked:
                    
                    weight_file  = os.path.join(self.run_dir, 'checkpoints',  f'checkpoint.{epoch:03d}.hdf5')

                    if not os.path.isfile(self.evaluation_file ) or epoch==0:
                        df = pd.DataFrame(columns=['epoch','accuracy'])
                        df.to_csv(self.evaluation_file )
                    else:
                        df = pd.read_csv(self.evaluation_file ,index_col=0)

                    print(self.pre, 'Restoring from '+str(weight_file))
                    self.model.load_weights(weight_file)

                    #############################################################################################
                    # INFER THE TEST SET
                    #############################################################################################
                    
                    for i, (X,Y) in enumerate(self.Data_Generator):
                        preds  = self.model.predict(X)
                        if i==0:
                            labels   = Y[0]
                            c_pred   = preds[0]
                            l_pred_w = preds[1][:,-1]
                            l_pred_s = preds[2][:,-1]
                        else :
                            labels   = np.concatenate([labels,Y[0]])  
                            c_pred   = np.concatenate([c_pred,preds[0]])
                            l_pred_w = np.concatenate([l_pred_w,preds[1][:,-1]])
                            l_pred_s = np.concatenate([l_pred_s,preds[2][:,-1]])


                    #############################################################################################
                    # GET VALUES
                    #############################################################################################
                    
                    pred_array = np.argmax(c_pred,axis=1)
                    annot_array = np.argmax(labels,axis=1)
                    scores_array = np.max(c_pred,axis=1)
                    correctness_array = (pred_array==annot_array).astype(np.int64)

                    #true_loss = list_np[2]
                    #pred_loss = list_np[3]

                    print(self.pre,"Length of the test: ", len(pred_array))

                    #############################################################################################
                    # COMPUTE METRICS
                    #############################################################################################

                    ######## Classification

                    # Compute the F1 score, also known as balanced F-score or F-measure
                    f1 = metrics.f1_score(annot_array, pred_array, average='macro')
                    self.wandb.log({'F1 score': f1}, step=epoch)

                    # Accuracy classification score
                    accuracy = metrics.accuracy_score(annot_array, pred_array)
                    self.wandb.log({'Test: Classification Accuracy': accuracy}, step=epoch)

                    # Compute Receiver operating characteristic (ROC)
                    false_positives_axe, true_positives_axe, roc_seuil = metrics.roc_curve(correctness_array, scores_array)
                    fig_roc = self.Plot_ROC(false_positives_axe, true_positives_axe, roc_seuil)
                    self.wandb.log({'Receiver operating characteristic (ROC)': fig_roc}, step=epoch)

                    #  Area Under the Curve (AUC) 
                    res_auc = metrics.auc(false_positives_axe, true_positives_axe)
                    self.wandb.log({'Area Under the Curve (AUC) ': res_auc}, step=epoch)  

                    # Compute confusion matrix to evaluate the accuracy of a classification.
                    if len(self.config["NETWORK"]["CLASSES"])<200:
                        cm = metrics.confusion_matrix(annot_array, pred_array).astype(np.float32)
                        fig_cm = self.Plot_confusion_matrix(cm)
                        self.wandb.log({'Confusion Matrix' : fig_cm}, step=epoch)

                    ######## Regression (loss estimation)
                    """
                    mae_v1 = metrics.mean_absolute_error(true_loss, pred_loss)
                    self.wandb.log({'Mean absolute error': mae_v1}, step=epoch)

                    evs_v1 = metrics.explained_variance_score(true_loss, pred_loss)
                    self.wandb.log({'Explained variance': evs_v1}, step=epoch)

                    mse_v1 = metrics.mean_squared_error(true_loss, pred_loss)
                    self.wandb.log({'Mean squared error': mse_v1}, step=epoch)
                    """

                    #############################################################################################
                    # Save accuracy and epoch to select best model
                    #############################################################################################
                    temp_df= pd.DataFrame({'epoch':[epoch],'accuracy':[accuracy]})

                    df = pd.concat([df,temp_df],ignore_index=True)

                    to_print = 'Test || '
                    to_print += "Epoch: %2d || "%(epoch)
                    to_print += "Accuracy: %.2f || "%(100*accuracy)

                    print(self.pre, to_print)
                # for

                df.to_csv(self.evaluation_file)

                  
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
