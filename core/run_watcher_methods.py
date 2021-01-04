from tensorflow.keras.callbacks import Callback

        
class Update_progress(Callback):
    def __init__(self,
                 run_watcher,
                 name_run,
                 steps_per_epoch, 
                 total_epochs, 
                 total_steps, 
                 current_epoch, 
                 current_step):
        
        self.run_watcher      = run_watcher
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

        
    def on_train_begin(self, logs=None):
        self.run_watcher.update_run.remote(name=self.name_run, status="Training")
        self.status["status"] = "Training"
        
    def on_epoch_end(self, epoch, logs=None):
        self.status["Current Epoch"]+=1
        
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
        self.status["status"] = "Finished"
        self.run_watcher.update_run.remote(name=self.name_run, status="Finished", progress=self.status["Progress"])