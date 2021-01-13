from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
        
class Update_progress(Callback):
    def __init__(self,
                 run_watcher,
                 wandb,
                 name_run,
                 steps_per_epoch, 
                 total_epochs, 
                 total_steps, 
                 current_epoch, 
                 current_step):
        
        self.run_watcher      = run_watcher
        self.wandb            = wandb
        self.name_run         = name_run        
        
        self.stop_flag = False

        self.divisor = int(total_steps/10000)
        if self.divisor==0:
            self.divisor=1
        
        self.status = {}
        self.status["Current Epoch"] = current_epoch+1
        self.status["Total Epoch"] = total_epochs
        self.status["Current Step"]  = current_step+1
        self.status["Total Steps"] = total_steps
        self.status["Progress"] = round(current_step  /total_steps  * 100.0, 3)
        self.status["steps_per_epoch"] = steps_per_epoch
        self.status["status"] = "Idle"
        
        self.translation_dict = {}
        self.translation_dict['loss']= "Loss: Total Loss"
        self.translation_dict['c_pred_loss']= "Loss: Classification Loss"
        self.translation_dict['l_pred_w_loss']= "Loss: Active Learning Loss"
        
        self.translation_dict['c_pred_sparse_categorical_accuracy']= "Metric: Train Classification Accuracy"
        #self.translation_dict['l_pred_w_MAE_Lossnet']= "Metric: Active Learning MAE"
        
        self.translation_dict['lr']= "Hyper: Learning rate"
        

    def on_train_begin(self, logs=None):
        self.run_watcher.update_run.remote(name=self.name_run, status="Training")
        self.status["status"] = "Training"
        
    def on_epoch_end(self, epoch, logs=None):
        for key in logs.keys():
            if key in self.translation_dict.keys():
                self.wandb.log({self.translation_dict[key]: logs[key]}, step=epoch)
                
        self.wandb.log({"Hyper: Whole training":  backend.get_value(self.model.loss_weights['l_pred_w'])}, step=epoch)
        self.wandb.log({"Hyper: Split training":  backend.get_value(self.model.loss_weights['l_pred_s'])}, step=epoch)
      
        self.status["Current Epoch"] = epoch
        
    def on_train_batch_end(self, batch, logs=None):
        
        self.status["Current Step"]+=1
        
        if (self.status["Current Step"] % self.divisor)==0:
            self.status["Progress"] =  round(self.status["Current Step"]  / self.status["Total Steps"]  * 100.0, 2)
            self.run_watcher.update_run.remote(name=self.name_run, progress=self.status["Progress"])
            
        if self.stop_flag:
            self.model.stop_training = True
            self.status["status"] = "Stopped"
            self.run_watcher.update_run.remote(name=self.name_run, status="Stopped", progress=self.status["Progress"])
            
    def on_train_end(self, logs=None):
        # set status run watcher
        self.status["status"] = "Finished"
        self.run_watcher.update_run.remote(name=self.name_run, status="Finished", progress=self.status["Progress"])