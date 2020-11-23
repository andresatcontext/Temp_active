import tensorflow as tf

def compute_loss(y_true, y_pred):
    margin=1.0
    reduction= 'mean'

    assert y_pred.shape[0] % 2 == 0, 'the batch size is not even.'
    
    #assert y_pred.shape == y_pred.flip(0).shape     
    # classification prediction 
    c_pred = y_pred[:,:-1]
    # loss prediction
    l_pred = y_pred[:,-1]
    l_pred_r = l_pred[::-1]
    assert l_pred.shape == l_pred_r.shape
    
    l_pred = (l_pred - l_pred_r)[:y_pred.shape[0]//2]
    
    # y_true is just the classification as the l_true is calculated here
    scc = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
    class_loss = scc(y_true,c_pred)
    
    target = (class_loss - class_loss[::-1])[:class_loss.shape[0]//2]
    target = tf.stop_gradient(target)
    
    one = (2 * tf.math.sign(  tf.clip_by_value( target, 0, 1))) - 1
    
    if reduction == 'mean':
        loss = tf.reduce_sum(tf.clip_by_value(margin - one * l_pred, 0,10000))
        loss = loss / tf.cast(l_pred.shape[0], tf.float64)  # Note that the size of l_pred is already halved
    elif reduction == 'none':
        loss = tf.clip_by_value(margin - one * l_pred, 0,10000)
    else:
        NotImplementedError()
    
    return loss


