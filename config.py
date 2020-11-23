import wandb
wandb.init(project="Active Learning")

#wandb.config.input_shape = (256,256,3) #x_train.shape[1:]
#wandb.config.classes_data = 70  #len(np.unique(y_train))
'''''
wandb.config.embedding_size = 128


wandb.config.margin = 1.0
wandb.config.reduction_in_loss = 'mean' # 'none'

wandb.config.w_classif_loss = 1.0
wandb.config.w_loss_loss = 0

wandb.config.batch_size = 128

wandb.config.epoch = 200
wandb.config.lr = 0.1
wandb.config.milestones = [160 ,200]
wandb.config.epochl = 120
wandb.config.momentum = 0.9
wandb.config.wdecay = 5e-4

'''''
# all the data not the loss prediction


wandb.config.embedding_size = 128


wandb.config.margin = 1.0
wandb.config.reduction_in_loss = 'mean' # 'none'

wandb.config.w_classif_loss = 1.0
wandb.config.w_loss_loss = 0

wandb.config.batch_size = 128

wandb.config.epoch = 50
wandb.config.lr = 0.1
wandb.config.milestones = [25, 35]
#wandb.config.epochl = 40
wandb.config.momentum = 0.9
wandb.config.wdecay = 5e-4
