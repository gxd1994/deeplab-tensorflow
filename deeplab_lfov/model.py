import tensorflow as tf
from six.moves import cPickle
import numpy as np

# Loading net skeleton with parameters name and shapes.
with open("./util/net_skeleton.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)

# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=21) -> [pixel-wise softmax loss].
num_layers    = [2, 2, 3, 3, 3, 1, 1, 1]
dilations     = [[1, 1],
                 [1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [2, 2, 2],
                 [12], 
                 [1], 
                 [1]]
n_classes = 2
# All convolutional and pooling operations are applied using kernels of size 3x3; 
# padding is added so that the output of the same size as the input.
ks = 3

def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation 
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.
    
    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """
    
    def __init__(self, weights_path=None):
        """Create the model.
        
        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """
        self.variables = self._create_variables(weights_path)
        
    def _create_variables(self, weights_path):
        """Create all variables used by the network.
        This allows to share them between multiple calls 
        to the loss function.
        
        Args:
          weights_path: the path to the ckpt file with dictionary of weights from .caffemodel. 
                        If none, initialise all variables randomly.
        
        Returns:
          A dictionary with all variables.
        """
        var = list()
        index = 0
        
        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f) # Load pre-trained weights.
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],
                                           name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and 
            # all biases to 0's.
            for name, shape in net_skeleton:
                if "/w" in name: # Weight filter.
                    w = create_variable(name, list(shape))
                    var.append(w)
                else:
                    b = create_bias_variable(name, list(shape))
                    var.append(b)
        return var
    
    
    def create_network(self,variables,input_batch, keep_prob):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.
          
        Returns:
          A downsampled segmentation mask. 
        """
        current = input_batch
        
        v_idx = 0 # Index variable.
        
        # Last block is the classification layer.
        for b_idx in xrange(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = variables[v_idx * 2]
                b = variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current, 
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        
        # Classification layer; no ReLU.
        w = variables[v_idx * 2]
        b = variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current


    def _create_network(self, input_batch, keep_prob):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.
          
        Returns:
          A downsampled segmentation mask. 
        """
        current = input_batch
        
        v_idx = 0 # Index variable.
        
        # Last block is the classification layer.
        for b_idx in xrange(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current, 
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        
        # Classification layer; no ReLU.
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current
    
    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.

        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.

        Returns:
          Outputs a tensor of shape [batch_size h w 21]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
            input_batch = tf.one_hot(input_batch, depth=2)
        return input_batch
      
    def preds(self, input_batch,variables):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        raw_output = self.create_network(variables,tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3,])
        raw_output = tf.argmax(raw_output, dimension=3)
        raw_output = tf.expand_dims(raw_output, dim=3) # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8)

    def _calculate_metrics(self,preds,labels):

        confusion_matrix = tf.contrib.metrics.confusion_matrix(labels,preds,num_classes = 2)
        confusion_matrix = tf.cast(confusion_matrix,tf.float32)

        recall1 = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
        recall2 = confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1])

        precision1 = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
        precision2 = confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])


        accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[0][0]+confusion_matrix[0][1])

        return recall1,recall2,precision1,precision2,accuracy


    def metrics(self,preds,labels):

        prediction = tf.reshape(preds, [-1])
        gt = tf.reshape(labels, [-1])

        recall1,recall2,precision1,precision2,accuracy = self._calculate_metrics(prediction,gt)


        tf.summary.scalar('val_r0',recall1,collections = ['val'])
        tf.summary.scalar('val_r1',recall2,collections = ['val'])
        tf.summary.scalar('val_p0',precision1,collections = ['val'])
        tf.summary.scalar('val_p1',precision2,collections = ['val'])
        tf.summary.scalar('val_acc',accuracy,collections = ['val'])


        return  recall1,recall2,precision1,precision2,accuracy

    # def create_network(self,img_batch, label_batch):
    #     self.variables()


    def loss(self, img_batch, label_batch, variables, is_Qnet = False, is_placehold = False):

        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """


        raw_output = self.create_network(variables,tf.cast(img_batch, tf.float32),keep_prob=tf.constant(0.5))

        # print raw_output

        prediction = tf.reshape(raw_output, [-1, n_classes])

        
        # Need to resize labels and convert using one-hot encoding.

        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])
        

        # Pixel-wise softmax loss.

        pred = tf.nn.softmax(prediction)

        train_metrics = self._calculate_metrics(tf.argmax(pred,axis=1),tf.argmax(gt,axis = 1))



        w_0 = 1.0
        w_1 = 1.0 


        tf.summary.scalar('w_0',w_0)
        tf.summary.scalar('w_1',w_1)

        w0_col = tf.multiply(w_0,gt[:,0])
        w1_col = tf.multiply(w_1,gt[:,1])


        weight_matrix = tf.stack([w0_col,w1_col],axis = 1)


        loss_tmp1 = -tf.multiply(weight_matrix,tf.log(tf.clip_by_value(pred,1e-5,1.0)))
        loss_tmp2 = tf.reduce_sum(loss_tmp1,axis = 1)
        loss_total = tf.reduce_mean(loss_tmp2)


        # loss_target = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = gt)
        # loss_target = tf.reduce_mean(loss_target)


        # tf.summary.scalar("loss_target",loss_target)

        if is_Qnet:
            tf.summary.scalar('train_r0',train_metrics[0],collections = ['Q_train'])
            tf.summary.scalar('train_r1',train_metrics[1],collections = ['Q_train'])
            tf.summary.scalar('train_p0',train_metrics[2],collections = ['Q_train'])
            tf.summary.scalar('train_p1',train_metrics[3],collections = ['Q_train'])
            tf.summary.scalar('train_acc',train_metrics[4],collections = ['Q_train'])
            tf.summary.scalar("loss",loss_total,collections = ['Q_train'])
            

        elif not is_placehold:
            tf.summary.scalar('train_r0',train_metrics[0],collections = ['train'])
            tf.summary.scalar('train_r1',train_metrics[1],collections = ['train'])
            tf.summary.scalar('train_p0',train_metrics[2],collections = ['train'])
            tf.summary.scalar('train_p1',train_metrics[3],collections = ['train'])
            tf.summary.scalar('train_acc',train_metrics[4],collections = ['train'])
            tf.summary.scalar("loss",loss_total,collections = ['train'])

        
        return loss_total,train_metrics,raw_output,label_batch

        
    
    def loss_w(self, img_batch, label_batch,variables, weight):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """


        raw_output = self.create_network(variables,tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))


        prediction = tf.reshape(raw_output, [-1, n_classes])

        
        # Need to resize labels and convert using one-hot encoding.

        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])
        

        # Pixel-wise softmax loss.

        pred = tf.nn.softmax(prediction)


        train_metrics = self._calculate_metrics(tf.argmax(pred,axis=1),tf.argmax(gt,axis = 1))




        w_0 = 1.0
        w_1 = weight


        # balanced_weihght = tf.get_variable('blanced_weighte',shape=[2],initializer = tf.constant_initializer(0))
        # balanced_weihght = tf.nn.softmax(balanced_weihght,name = 'softmax_balanced_weihght')


        # tf.summary.scalar('balanced_weihght_0',balanced_weihght[0])
        # tf.summary.scalar('balanced_weihght_1',balanced_weihght[1])

        # count_0 = tf.cast(tf.reduce_sum(gt[:,0]),tf.float32)
        # count_1 = tf.cast(tf.reduce_sum(gt[:,1]),tf.float32)

        # w_0 = balanced_weihght[0]
        # w_1 = balanced_weihght[1] * count_0 / count_1


        # count_0 = tf.cast(tf.reduce_sum(gt[:,0]),tf.float32)
        # count_1 = tf.cast(tf.reduce_sum(gt[:,1]),tf.float32)

        # w_0 = 1.0
        # w_1 = count_0/count_1

        tf.summary.scalar('w_0',w_0)
        tf.summary.scalar('w_1',w_1)

        w0_col = tf.multiply(w_0,gt[:,0])
        w1_col = tf.multiply(w_1,gt[:,1])


        weight_matrix = tf.stack([w0_col,w1_col],axis = 1)



        # weight_matrix = tf.Print(weight_matrix,[tf.shape(weight_matrix),weight_matrix],message= 'weight_matrix',summarize = 6)


        loss_tmp1 = -tf.multiply(weight_matrix,tf.log(tf.clip_by_value(pred,1e-5,1.0)))
        loss_tmp2 = tf.reduce_sum(loss_tmp1,axis = 1)
        loss_total = tf.reduce_mean(loss_tmp2)




        # recall,op1 = tf.metrics.recall(tf.argmax(gt,axis=1),tf.argmax(pred,axis=1))

        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(gt,axis=1),tf.argmax(pred,axis=1)),tf.float32))
        
        # precision,op2 = tf.metrics.precision(tf.argmax(gt),tf.argmax(prediction))

        # accuracy,op3 = tf.metrics.accuracy(tf.argmax(gt),tf.argmax(prediction))


        # metrics = [precision,recall,accuracy]



        # loss_target = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = gt)
        # loss_target = tf.reduce_mean(loss_target)

        # tf.summary.scalar("loss_target",loss_target)


        
        return loss_total,train_metrics,raw_output








# i0 = tf.constant(0)
# m0 = tf.zeros([1,2])

# loop_var = [i0,m0]
# c = lambda i,m: i < gt.get_shape()[0].value

# def body(i,m):
#   cond = tf.equal(gt[i][0],1)
#   f1 = lambda : tf.convert_to_tensor([[w0+1,0]],dtype=tf.float32)
#   f2 = lambda : tf.convert_to_tensor([[0,w1+1]],dtype=tf.float32)
#   w = tf.cond(cond,f1,f2)
#   op = tf.concat([m,w],axis=0)
#   i = i + 1
#   return i,op

# i_,weight_matrix = tf.while_loop(c,body,loop_var,shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])

# def my_func(label,w0,w1):
#   num = label.shape[0]
#   # print "label.shape[0]",num
#   print "w0,w1",w0,w1
#   m = np.zeros_like(label,dtype = np.float32)
#   for i in range(num):
#     if label[i][0] == 1:
#       m[i][0] = w0+1
#       m[i][1] = 0
#     else:
#       m[i][0] = 0
#       m[i][1] = w1+1
#   return m

# weight_matrix = tf.py_func(my_func,[gt,w0,w1],tf.float32)

# weight_matrix = tf.Print(weight_matrix,[tf.shape(weight_matrix),weight_matrix],message= 'weight_matrix',summarize = 10000)


        # gt = tf.Print(gt,[tf.shape(gt),gt],message= 'weight_matrix',summarize = 6)
        # weight_matrix = tf.Print(weight_matrix,[tf.shape(weight_matrix[1:,:]),weight_matrix],message= 'weight_matrix',summarize = 6)

        # with tf.control_dependencies([gt,weight_matrix]):
          # loss_tmp = -tf.multiply(weight_matrix[1:,:],tf.log(tf.clip_by_value(pred,1e-5,1.0)))
          # loss_tmp = tf.reduce_sum(loss_tmp,axis = 1)
          # loss_total = tf.reduce_mean(loss_tmp)