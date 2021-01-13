from AutoML import DataNet, AutoMLDataset, local_module, local_file, AutoML, get_run_watcher, get_user
import ray

@ray.remote(num_gpus=1)
class Active_Learning_test:
    def __init__(self,   config, 
                         dataset,
                         num_run):

        
        #############################################################################################
        # LIBRARIES
        #############################################################################################        
        import os
        import numpy as np
        import tensorflow as tf
        from PIL import ImageFont
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.keras import optimizers, losses, models, backend, layers, metrics
        
        self.run_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.run_path)
        
        utils         = local_module("utils")
        logger        = local_module("logger")
        lossnet       = local_module("lossnet")
        data_pipeline = local_module("data_pipeline")
        backbones     = local_module("backbones")
        
        self.font_size = 12
        self.font = ImageFont.truetype(os.path.join(self.run_path,"miscellaneous","InconsolataGo-Nerd-Font-Complete-Mono.ttf"), size=self.font_size)
        
        #############################################################################################
        # PARAMETERS RUN
        #############################################################################################
        self.number_images_vid  = 30
        
        self.config          = config
        self.dataset         = dataset
        self.num_run         = num_run
        self.group           = "Stage_"+str(num_run)
        self.name_run        = "Test"+self.group 
        
        self.run_dir         = os.path.join(config["PROJECT"]["group_dir"],self.group)
        self.run_dir_check   = os.path.join(self.run_dir ,'checkpoints')
        self.checkpoints_path= os.path.join(self.run_dir_check,'checkpoint.{epoch:03d}.hdf5')
        self.user            = get_user()
        
        self.transfer_weight_path = self.config['TRAIN']["transfer_weight_path"]
        self.input_shape       = [self.config["NETWORK"]["INPUT_SIZE"], self.config["NETWORK"]["INPUT_SIZE"], 3]
        
        self.pre ='\033[1;36m' + self.name_run + '_' + config["PROJECT"]["group"]  + '\033[0;0m' #"____" #
        self.problem ='\033[1;31m' + self.name_run+ '_' + config["PROJECT"]["group"] + '\033[0;0m'
        
        # Creating the test folder
        self.evaluation_folder = os.path.join(self.run_dir, 'evaluation')
        self.evaluation_file   = os.path.join(self.evaluation_folder, "accuracy_epoch.csv")
        
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
                # SETUP WANDB
                #############################################################################################
                import wandb

                self.wandb = wandb
                self.run_wandb = self.wandb.init(project  = config["PROJECT"]["project"], 
                                                group    = config["PROJECT"]["group"], 
                                                name     = "Test_"+str(num_run),
                                                job_type = self.group ,
                                                sync_tensorboard = True,
                                                config = config)

                #############################################################################################
                # LOAD DATA
                #############################################################################################
                self.DataGen = data_pipeline.ClassificationDataset( config["TEST"]["batch_size"],
                                                                    self.dataset,
                                                                    subset = "test",
                                                                    original_size      = config["DATASET"]["original_size"],
                                                                    data_augmentation  = False)  
                
                
                self.num_class = len(self.DataGen.list_classes)
                
                #############################################################################################
                # GLOBAL PROGRESS
                #############################################################################################
                self.steps_per_epoch  = int(np.ceil(self.DataGen.nb_elements/config["TEST"]["batch_size"]))
                self.total_epochs  = self.config['TRAIN']["EPOCH_WHOLE"] + self.config['TRAIN']["EPOCH_SLIT"]
                self.total_tests = int(np.floor(self.total_epochs/self.config['TRAIN']["test_each"]))
                print(self.pre,'Number of elements in the test set', self.DataGen.nb_elements)
                
                
                #############################################################################################
                # DEFINE CLASSIFIER
                #############################################################################################
                # set input
                img_input = tf.keras.Input(tensor=self.DataGen.images_tensor ,name= 'input_image')
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
                # INIT VARIABLES
                #############################################################################################
                #self.sess.graph.as_default()
                backend.set_session(self.sess)
                self.sess.run(tf.local_variables_initializer())

                ##################
                # SETUP WATCHER
                ################## 
                self.run_watcher = get_run_watcher()

                self.run_watcher.add_run.remote(name=self.name_run,
                                                user=self.user,
                                                progress=0,
                                                wandb_url=self.wandb.run.get_url(),
                                                status="Idle") 
                
                self.progress = 0
                
                self.run_watcher.update_run.remote(name=self.name_run, progress=self.progress)
                
                print(self.pre,'Init done')

    @ray.method(num_returns = 0)
    def evaluate(self):
        import numpy as np
        from sklearn import metrics
        import pandas as pd
        import os
        import time
        
        
        with self.graph.as_default():
            with self.sess.as_default():
                print( self.pre ,"Start testing service")
                epochs_checked = []
                
                self.run_watcher.update_run.remote(name=self.name_run, status="Testing Service")
                
                while True:
                    while True:
                        read_folder = [int(i.split('.')[-2]) for i in os.listdir(self.run_dir_check) if i.endswith('.hdf5')]
                        new_epochs = list(set(read_folder) - set(epochs_checked))
                        time.sleep(5)
                        if new_epochs:
                            new_epochs.sort()
                            break
                        
                    
                    for epoch in new_epochs:

                        if not os.path.isfile(self.evaluation_file ) or epoch==1:
                            df = pd.DataFrame(columns=['epoch','accuracy'])
                            df.to_csv(self.evaluation_file )
                        else:
                            df = pd.read_csv(self.evaluation_file ,index_col=0)

                        #############################################################################################
                        # LOAD PATH TO TEST
                        #############################################################################################
                        model_path = os.path.join(self.run_dir_check,f'checkpoint.{epoch:03d}.hdf5')
                        print(self.pre, "Loading weigths from: ",model_path)
                        self.model.load_weights(model_path)
                        
                        # set the dataset to the beggining
                        self.sess.run(self.DataGen.iterator.initializer)
                        

                        #############################################################################################
                        # INFER TEST SET
                        #############################################################################################
                        results = self.model.predict(None,steps=self.steps_per_epoch)

                        print('Check number of different files: ',len(set(results[-1])))

                        #############################################################################################
                        # GET VALUES
                        #############################################################################################
                        # c_pred_1, loss_pred_embeddings[0], loss_pred_embeddings[2], labels_tensor, files_tesor]
                        pred_array = np.argmax(results[0],axis=1)
                        annot_array = np.squeeze(results[3])
                        scores_array = np.max(results[0],axis=1)
                        correctness_array = (pred_array==annot_array).astype(np.int64)

                        files_names = results[4]
                        pred_loss    = results[1][:,-1]
                        
                        # generate dataframe to easy manipulate the data
                        to_df ={}
                        #to_compu['s_uncertanty'] = min_max_scaler.fit_transform(np_list[2].reshape(-1, 1)).squeeze()
                        to_df['uncertanty'] = pred_loss
                        to_df['files']      = files_names
                        to_df['scores']     = scores_array
                        to_df['prediction'] = pred_array
                        to_df['labels']     = annot_array
                        
                        df_res =  pd.DataFrame(to_df)
                        df_res['label_2'] = [self.DataGen.real_names_classes[i] for i in df_res.labels]
                        df_res['pred']    = [self.DataGen.real_names_classes[i] for i in df_res.prediction]
                        path_save_res = os.path.join(self.evaluation_folder, f"results_test_{epoch:03d}.csv")
                        df_res.to_csv(path_save_res)
                        
                        df_res.sort_values('scores',inplace=True)
                        df_res.reset_index(drop=True,inplace=True)
                       
                        
                        # check if len test is ok
                        print(self.pre,"Length of the test: ", len(pred_array))
                        print(self.pre,'First and last file: ',files_names[0],files_names[-1])

                        #############################################################################################
                        # COMPUTE METRICS
                        #############################################################################################

                        ######## Classification

                        # Compute the F1 score, also known as balanced F-score or F-measure
                        f1 = metrics.f1_score(annot_array, pred_array, average='macro')
                        self.wandb.log({'Metric: Test F1 score': f1}, step=epoch)

                        # Accuracy classification score
                        accuracy = metrics.accuracy_score(annot_array, pred_array)
                        self.wandb.log({'Metric: Test Classification Accuracy': accuracy}, step=epoch)

                        # Compute Receiver operating characteristic (ROC)
                        false_positives_axe, true_positives_axe, roc_seuil = metrics.roc_curve(correctness_array, scores_array)
                        fig_roc = self.Plot_ROC(false_positives_axe, true_positives_axe, roc_seuil)
                        self.wandb.log({'Metric: Test Receiver operating characteristic (ROC)': fig_roc}, step=epoch)

                        #  Area Under the Curve (AUC) 
                        res_auc = metrics.auc(false_positives_axe, true_positives_axe)
                        self.wandb.log({'Metric: Test Area Under the Curve (AUC) ': res_auc}, step=epoch)  

                        # Compute confusion matrix to evaluate the accuracy of a classification.
                        if len(self.DataGen.list_classes)<200:
                            cm = metrics.confusion_matrix(annot_array, pred_array).astype(np.float32)
                            fig_cm = self.Plot_confusion_matrix(cm)
                            self.wandb.log({'Metric: Test Confusion Matrix' : fig_cm}, step=epoch)
                        
                        if (self.total_epochs-4)<=epoch:
                            if len(self.DataGen.list_classes)<20:
                                self.generate_videos(df_res,epoch)


                        #############################################################################################
                        # Save accuracy and epoch to select best model
                        #############################################################################################
                        temp_df= pd.DataFrame({'epoch':[epoch],'accuracy':[accuracy]})

                        df = pd.concat([df,temp_df],ignore_index=True)

                        to_print = 'Test || '
                        to_print += "Epoch: %2d || "%(epoch)
                        to_print += "Accuracy: %.2f || "%(100*accuracy)

                        print(self.pre, to_print)

                        df.to_csv(self.evaluation_file)
                    
                        epochs_checked+=[epoch]
                        
                        self.progress = len(epochs_checked)/self.total_tests
                        
                        self.run_watcher.update_run.remote(name=self.name_run, progress=self.progress)
                    
                    if len(epochs_checked)==self.total_tests:
                        break
                        
                
                time.sleep(30)    
                #self.run_wandb.finish()
                self.run_watcher.update_run.remote(name=self.name_run, status="End")
                print( self.pre ,"End testing service")
                
    @ray.method(num_returns = 1)
    def get_progress(self):
        return self.progress
                
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
                                'x': self.DataGen.real_names_classes,
                                'y': self.DataGen.real_names_classes,
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

    def generate_videos(self,df_res,epoch):
        # pil for maniupulating the images from datanet
        from PIL import Image,ImageFont,ImageDraw
        import numpy as np
        import time
        # process images
        def get_image(info):
            
         
            filename = str(info['files'])

            if "'" in filename:
                filename = filename.split("'")[1]


            im = self.DataGen.datanet.get_image(filename)
            desired_size=256

            old_size = im.size  # old_size[0] is in (width, height) format

            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            im = im.resize(new_size, Image.ANTIALIAS)

            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size-new_size[0])//2,
                                (desired_size-new_size[1])//2))

            draw = ImageDraw.Draw(new_im)
            
            i = 0
            draw.text((0, i*self.font_size),f"Label {info['label_2']}",(255,0,255),font=self.font)
            i +=1
            draw.text((0, i*self.font_size),f"Pred  {info['pred']}",(255,0,255),font=self.font)
            i +=1
            draw.text((0, i*self.font_size),f"Score {np.round(info['scores'],2)}",(255,0,255),font=self.font)

            x = np.array(new_im)
            x = np.expand_dims(np.moveaxis(x, -1, 0),0)

            return x

        # for each class divide the min and max score to sample images with different scores
        for cls_i in range(len(self.DataGen.list_classes)):
            
            temp_df_res = df_res[df_res.labels==cls_i]
            #print(temp_df_res.describe())
            
            #print(50*"_")
            min_score   = temp_df_res.scores.min()
            max_score   = temp_df_res.scores.max()
            delta_score = (max_score-min_score)/(2*self.number_images_vid)
            list_images = []
            for delta_i in range(self.number_images_vid):  
                score_i = min_score+0.001+(delta_i*delta_score)
                #print(score_i)
                #print(50*"_")
                info_df = temp_df_res[temp_df_res.scores<=score_i].iloc[-1]
                #print(info_df)
                #print(50*"_")
                list_images.append(get_image(info_df))
                
            print(len(list_images))
            print(np.concatenate(list_images).shape)
            
            self.wandb.log({"Sample: "+self.DataGen.real_names_classes[cls_i] : self.wandb.Video(np.concatenate(list_images), fps=1, format="mp4")},step=epoch)
            time.sleep(5)
                
        # generate videos posible bad annotations (False positifs)
        temp_df_res = df_res[df_res.prediction!=df_res.labels].iloc[-self.number_images_vid:]
        list_images = []
        for info_df in temp_df_res.iterrows():
            list_images.append(get_image(info_df[1]))
            
        self.wandb.log({"Sample: False Positifs" : self.wandb.Video(np.concatenate(list_images), fps=1, format="mp4")},step=epoch)
        time.sleep(5)

            
                
            
            
        
            
        





