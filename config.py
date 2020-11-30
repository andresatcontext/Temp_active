
def Load_config(NUM_TRAIN,input_shape,num_class,active_learning = False):
    # just to select faster if run active learning or train the whole network
    active_learning = False

    # total images for training
    #config["NUM_TRAIN"] = len(x_train) # N
    ## shape input
    #config["input_shape"] = x_train.shape[1:]
    config = {}
    
    # total images for training
    config["NUM_TRAIN"] = NUM_TRAIN# len(x_train) # N
    # shape input
    config["input_shape"] = input_shape# x_train.shape[1:]
    
    
    # parametres data augmentation
    config["width_shift_range"] = 4
    config["height_shift_range"] = 4
    config["horizontal_flip"] = True

    # number of classes
    config["num_class"] = num_class# len(np["unique(y_train))

    # common config
    # length embedding for z
    config["embedding_size"] = 128
    config["batch_size"] = 64
    
    
    # Training parameters:
    config["start_epoch"] = 0
    

    # 

    if active_learning:

        # how many images infer to check its importance to training
        config["SUBSET"] = 10000 # M
        # from the subset select the best ADDENDUM images to train the network
        config["ADDENDUM"] = 1000 # K

        # How many times train test the algorithm with different starting points (different data to train 0)
        config["TRIALS"] = 1
        # for how many cycles for annotation time (CYCLES*ADDENDUM"] = total labeled images)
        config["CYCLES"] = 10
        
        # Warm up epochs
        config["EPOCH_WARMUP"] = 2
        # whole training epochs
        config["EPOCH_WHOLE"] = 120
        # split training epochs
        config["EPOCH_SLIT"] = 80 

        # change learning rate after the numbers in the list
        config["MILESTONES"] = [160]

        # TODO this has not been implemented
        config["EPOCHL"] = 120 
        # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

        # Parameters losses
        # "learning loss" loss
        config["MARGIN"] = 1.0
        config["reduction"] ='mean' # 'none'
        
        # Parametes optimizer
        # learning rate
        config["lr"] = 1e-1
        # Gamma for MultiStepLR
        config["gamma"] = 1e-1
        # 
        config["momentum"] = 0.9
        config["wdecay"] = 5e-4

        # weights when adding the losses
        config["w_c_loss"] = 1.0
        config["w_l_loss"] = 1.0

    else:

        # how many images infer to check its importance to training
        config["SUBSET"] = config["NUM_TRAIN"]
        # from the subset select the best ADDENDUM images to train the network
        config["ADDENDUM"] = config["NUM_TRAIN"] # K

        # How many times train test the algorithm with different starting points (different data to train 0)
        config["TRIALS"] = 1
        # for how many cycles for annotation time (CYCLES*ADDENDUM"] = total labeled images)
        config["CYCLES"] = 1

        # Warm up epochs
        config["EPOCH_WARMUP"] = 2
        # whole training epochs
        config["EPOCH_WHOLE"] = 40
        # split training epochs
        config["EPOCH_SLIT"] = 10

        # change learning rate after the numbers in the list
        config["MILESTONES"] = [25, 35]

        # TODO this has not been implemented
        config["EPOCHL"] = 40 


        # Parameters losses
        # "learning loss" loss
        config["MARGIN"] = 1.0
        config["reduction"] ='mean' # 'none'
        
        # Parametes optimizer
        # learning rate
        config["lr"] = 1e-5
        # Gamma for MultiStepLR
        config["gamma"] = 1e-1
        # 
        config["momentum"] = 0.9
        config["wdecay"] = 5e-4

        # weights when adding the losses
        config["w_c_loss"] = 1.0
        config["w_l_loss"] = 0.0
    return config
