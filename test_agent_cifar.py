import ray

@ray.remote(num_gpus=1)
class AL_test_cifar:
    def __init__(self,  project, 
                        source, 
                        dataset_name, 
                        group, 
                        run_dir, 
                        datanet, 
                        dataset_header, 
                        config, 
                        weight_file, 
                        step, 
                        wandb_id=None, 
                        agnostic_eval=False):
        
        
        import os
        self.run_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.run_path)
        
        from AutoML import AutoMLDataset, local_module
        import tensorflow as tf
        
        core = local_module("core")

        backbones = local_module("backbones")

        from tensorflow.python import pywrap_tensorflow
        import numpy as np
        
        self.backbone = backbones.ResNet18
        
        
        #############################################################################################
        # PARAMETERS 
        #############################################################################################
        self.project        = project
        self.source         = source
        self.dataset_name   = dataset_name
        self.group          = group
        self.run_dir        = run_dir
        self.datanet        = datanet
        self.dataset_header = dataset_header
        self.step           = step
        
        self.weight_name       = weight_file
        self.weight_file       = os.path.join(self.run_dir, 'checkpoint', weight_file)
        self.evaluation_folder = os.path.join(self.run_dir, 'evaluation')
        
        
        # change value from the config to not reduce the losses in order to compute the metrics for LossNet
        config['reduction'] = 'none'

        try:
            os.mkdir(self.evaluation_folder)
        except:
            pass
        
        #############################################################################################
        # SETUP WANDB
        #############################################################################################
        import wandb
        self.wandb = wandb

        if wandb_id is None:
            self.wandb.init(project=self.project, 
                group=self.group, 
                job_type="test",
                config=config)
            self.run_id = wandb.run.id
        else:
            self.wandb.init(project=self.project, 
                group=self.group, 
                job_type="test",
                id=wandb_id, 
                resume=True,
                config=config)
            self.run_id = wandb_id
            
        self.config = self.wandb.config

        
        #############################################################################################
        # LOAD DATA
        #############################################################################################
        if self.source=='CIFAR':
            from data_utils import CIFAR10Data
            # Load data
            cifar10_data = CIFAR10Data()
            num_classes = len(cifar10_data.classes)
            _, _, x_test, y_test = cifar10_data.get_data(subtract_mean=False)
            
            self.classes = cifar10_data.classes
        else:
            pass
        
        #############################################################################################
        # DATA GENERATOR
        #############################################################################################
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow(x_test,
                                       y_test,
                                       batch_size=self.config.batch_size,
                                       shuffle=False)
        
        features_shape = [None, 32, 32, 3]
        labels_shape = [None, 10]
        
        tf_data = tf.data.Dataset.from_generator(lambda: train_gen, 
                                                 output_types=(tf.float32, tf.float32),
                                                output_shapes = (tf.TensorShape(features_shape), tf.TensorShape(labels_shape)))
        data_tensors = tf_data.make_one_shot_iterator().get_next()
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
            self.model = core.Classifier_AL(self.backbone, self.config)
            self.c_pred, self.l_pred_w, self.l_pred_s = self.model.build_nework(self.img_input)

            self.c_loss, self.l_loss_w, self.l_loss_s, self.l_true = self.model.compute_loss(self.c_true)
        
            

        #############################################################################################
        # GLOBAL PROGRESS
        #############################################################################################
        self.evaluation_steps = int(np.ceil(len(x_test) / self.config.batch_size))


        #############################################################################################
        # DEFINE WEIGHT DEACY
        #############################################################################################
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.config.wdecay)
           
        #############################################################################################
        # METRICS
        ############################################################################################# 
        with tf.name_scope("define_metrics"):
            
            with tf.name_scope('Categorical_Accuracy'):
                correct_prediction = tf.equal( tf.argmax(self.c_true, 1), tf.argmax(self.c_pred, 1))
                self.Categorical_Accuracy = tf.cast(correct_prediction, tf.float32)
                
            with tf.name_scope('MAE_learning_loss_whole'):
                self.MAE_whole = tf.math.abs(tf.math.subtract(self.model.class_loss_non_reducted, self.l_loss_w))

            with tf.name_scope('MAE_learning_loss_split'):
                self.MAE_split = tf.math.abs(tf.math.subtract(self.model.class_loss_non_reducted, self.l_loss_s))
                
                
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
        import time
        import plotly.graph_objects as go
        from sklearn import metrics
        
        
        #############################################################################################
        # CREATE FOLDERS TO SAVE THE DATA
        #############################################################################################
        self.evaluation_folder = os.path.join(self.evaluation_folder, self.weight_name)
        if os.path.exists(self.evaluation_folder): shutil.rmtree(self.evaluation_folder)
        os.mkdir(self.evaluation_folder)

        predicted_dir_path = os.path.join(self.evaluation_folder, 'predicted')
        ground_truth_dir_path = os.path.join(self.evaluation_folder, 'ground-truth')
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(predicted_dir_path)
        
        #############################################################################################
        # INFER THE TEST SET
        #############################################################################################
        list_np = []
        for step in range(self.evaluation_steps):

            result_step= self.sess.run([self.c_pred,
                                        self.c_true,
                                        self.model.class_loss_non_reducted,
                                        self.l_pred_w,
                                        self.l_true,
                                        self.MAE_whole])
            
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
        
        true_loss_v1 = list_np[2]
        pred_loss    = list_np[3]
        true_loss_v2 = list_np[4]
        
        #############################################################################################
        # COMPUTE METRICS
        #############################################################################################
        
        ######## Classification
        
        # Compute the F1 score, also known as balanced F-score or F-measure
        f1 = metrics.f1_score(annot_array, pred_array, average='macro')
        self.wandb.log({'F1 score': f1}, step=self.step)
        
        # Accuracy classification score.
        accuracy = metrics.accuracy_score(annot_array, pred_array)
        self.wandb.log({'Accuracy classification score': accuracy}, step=self.step)
        
        # Compute Receiver operating characteristic (ROC)
        false_positives_axe, true_positives_axe, roc_seuil = metrics.roc_curve(correctness_array, scores_array)
        fig_roc = self.Plot_ROC(false_positives_axe, true_positives_axe, roc_seuil)
        self.wandb.log({'Receiver operating characteristic (ROC)': fig_roc}, step=self.step)
        
        #  Area Under the Curve (AUC) 
        res_auc = metrics.auc(false_positives_axe, true_positives_axe)
        self.wandb.log({'Area Under the Curve (AUC) ': res_auc}, step=self.step)  
        
        # Compute confusion matrix to evaluate the accuracy of a classification.
        if self.config.num_class<200:
            cm = metrics.confusion_matrix(annot_array, pred_array).astype(np.float32)
            fig_cm = self.Plot_confusion_matrix(cm)
            self.wandb.log({'Confusion Matrix' : fig_cm}, step=self.step)
                   
        ######## Regression (loss estimation)
        
        mae_v1 = metrics.mean_absolute_error(true_loss_v1, pred_loss)
        self.wandb.log({'Mean absolute error v1': mae_v1}, step=self.step)
        mae_v2 = metrics.mean_absolute_error(true_loss_v2, pred_loss)
        self.wandb.log({'Mean absolute error v2': mae_v2}, step=self.step)
        
        evs_v1 = metrics.explained_variance_score(true_loss_v1, pred_loss)
        self.wandb.log({'Explained variance v1': evs_v1}, step=self.step)
        evs_v2 = metrics.explained_variance_score(true_loss_v2, pred_loss)
        self.wandb.log({'Explained variance v2': evs_v2}, step=self.step)
                
        mse_v1 = metrics.mean_squared_error(true_loss_v1, pred_loss)
        self.wandb.log({'Mean squared error v1': mse_v1}, step=self.step)
        mse_v2 = metrics.mean_squared_error(true_loss_v2, pred_loss)
        self.wandb.log({'Mean squared error v2': mse_v2}, step=self.step)          

                  
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
                                'x': self.classes,
                                'y': self.classes,
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
