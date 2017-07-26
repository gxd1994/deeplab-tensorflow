"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time,shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

import math

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels

# import deep_q_network as DQN

import DDPG_update as DDPG

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


NUM_CLASS = 2

BATCH_SIZE = 6

# DATA_DIRECTORY = './dataset/VOC2012/VOC2012' #'/home/VOCdevkit'
DATA_DIRECTORY = './dataset/class1_def' #'/home/VOCdevkit'

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN = np.array((70,70,70), dtype=np.float32)


DATA_TRAIN_LIST_PATH = DATA_DIRECTORY+'/train.txt'
DATA_VAL_LIST_PATH = DATA_DIRECTORY+'/test.txt'
# DATA_VAL_LIST_PATH = DATA_DIRECTORY+'/val.txt'

INPUT_SIZE = '505,505'
LEARNING_RATE = 0.0001
NUM_STEPS = 60000
RANDOM_SCALE = False  #True
RESTORE_FROM = None #'./snapshots/model.ckpt-5000' # './snapshots/model.ckpt-pretrained'   #'./deeplab_lfov.ckpt'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = './util/net_weights.ckpt'
LOG_DIR = './log'

# VAL_LOOP = int(math.ceil(float(1449)/BATCH_SIZE))
VAL_LOOP = int(math.ceil(float(29)/BATCH_SIZE))



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_train_list", type=str, default=DATA_TRAIN_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_val_list", type=str, default=DATA_VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")

    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                       help="where to save log file")

    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    
def load(loader, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def copy_tensor_list(tensor_list):
	num = len(tensor_list)
	result_list = []
	for i in range(num):
		tmp = tf.Variable(0.0)
		tmp = tf.assign(tmp,tensor_list[i],validate_shape = False)
		result_list.append(tmp)

	return result_list

def updata_fun(variables,tag_variables):

	variables_copy = copy_tensor_list(variables)

	num = len(variables)

	for i in range(num):
		variables_copy[i] = tf.assign(variables_copy[i],tag_variables[i])

	return variables_copy


def TensorList_Assign(sources,tags):
    num = len(sources)
    ops =[]
    for i in range(num):
        op = tf.assign(tags[i],sources[i])
        ops.append(op)
    return ops

def calculate_state(label,featuremap,step,distribution,distribution_batch,is_batch_state):


    if is_batch_state:

        recall = recall_score(y_true = label,y_pred = featuremap,labels = [i for i in range(NUM_CLASS)],average=None)
        precision = precision_score(y_true = label,y_pred = featuremap,labels = [i for i in range(NUM_CLASS)],average=None)

        state = np.stack((distribution,distribution_batch,recall,precision),axis = 1).flatten()

        print('recall','precision','distribution','distribution_batch',recall,precision,distribution,distribution_batch)
        
        #print('state',state.shape,state)

    else:

        batch_total = label.shape[0]
        for i in range(distribution.shape[0]):
            distribution_batch[i] = 1.0 * np.sum(label == i)/batch_total

        # print('distribution_batch',distribution_batch)

        distribution = distribution + 1.0/step * (distribution_batch - distribution)


        recall = recall_score(y_true = label,y_pred = featuremap,labels = [i for i in range(NUM_CLASS)],average=None)
        precision = precision_score(y_true = label,y_pred = featuremap,labels = [i for i in range(NUM_CLASS)],average=None)


        state = np.stack((distribution,distribution_batch,recall,precision),axis = 1).flatten()

        print('recall','precision','distribution','distribution_batch',recall,precision,distribution,distribution_batch)
        
        #print('state',state.shape,state)

    return state,distribution,distribution_batch




# def Action_Assign(num):
#     action_type = [0.1,1.0,10]
#     action_list= []
#     for i in range(num):
#         for j in range(len(action_type)):
#             action_list.append()

def main():
    """Create the model and start the training."""
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_train_inputs"):
        reader_train = ImageReader(
            args.data_dir,
            args.data_train_list,
            input_size,
            RANDOM_SCALE,
            IMG_MEAN,
            coord)
        image_batch_train, label_batch_train = reader_train.dequeue(args.batch_size)

    with tf.name_scope("create_val_inputs"):
        reader_val = ImageReader(
            args.data_dir,
            args.data_val_list,
            input_size,
            False,
            IMG_MEAN,
            coord)
        image_batch_val, label_batch_val = reader_val.dequeue(args.batch_size, is_training = False)

    is_training = tf.placeholder(tf.bool,shape = [],name = 'stauts')


    image_batch,label_batch = tf.cond(is_training,lambda: (image_batch_train,label_batch_train),lambda: (image_batch_val,label_batch_val))



    # Q_SegNet input

    Q_image_batch = tf.placeholder(tf.float32,[BATCH_SIZE,505,505,3])
    Q_label_batch = tf.placeholder(tf.uint8,[BATCH_SIZE,505,505,1])

    loss_image_batch = tf.placeholder(tf.float32,[BATCH_SIZE,505,505,3])
    loss_label_batch = tf.placeholder(tf.uint8,[BATCH_SIZE,505,505,1])

    # DQN action  >>>>> weight

    Q_action = tf.placeholder(tf.float32,shape=[NUM_CLASS],name = 'q_action')


    # Create network.

    with tf.variable_scope('SegNet'):
        net = DeepLabLFOVModel(args.weights_path)


    with tf.variable_scope('Q_SegNet'):
        Q_net = DeepLabLFOVModel(args.weights_path)



    updata_q_net_to_net_ops = TensorList_Assign(Q_net.variables,net.variables)
    updata_net_to_q_net_ops = TensorList_Assign(net.variables,Q_net.variables)

    # net_variables_final = tf.cond(updata_q_net_to_net,lambda: updata_fun(net.variables,Q_net.variables),lambda: net.variables)
    # Q_net_variables_final = tf.cond(updata_net_to_q_net,lambda: updata_fun(Q_net.variables,net.variables),lambda: Q_net.variables)



    # Define Q_SegNet the loss 

    Q_loss_w,Q_train_metrics_w,Q_featuremap_w,_ = Q_net.loss_w(Q_image_batch, Q_label_batch, Q_net.variables, Q_action)

    Q_loss,Q_train_metrics,Q_featuremap,_,_ = Q_net.loss(Q_image_batch, Q_label_batch, Q_net.variables,is_Qnet = True)

    
    # Define SegNet the loss

    loss,train_metrics,_ ,_,train_updata_op= net.loss(image_batch, label_batch, net.variables)

    newest_loss,_,featuremap_get,label_equal,_ = net.loss(loss_image_batch,loss_label_batch,net.variables,is_placehold = True)


    # optimisation

    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    trainable = tf.trainable_variables()


    with tf.control_dependencies([loss]):
        optim = optimiser.minimize(loss,var_list = net.variables)   #, var_list=trainable)

    with tf.control_dependencies([Q_loss_w]):
        Q_optim = optimiser.minimize(Q_loss_w,var_list = Q_net.variables)   #, var_list=trainable)


    # SegNet validation

    pred = net.preds(image_batch,net.variables)

    val_metrics,val_updata_op = net.metrics(pred,label_batch)



    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8


    # Saver for storing checkpoints of the model.
    #saver = tf.train.Saver(var_list=trainable, max_to_keep=40)
    saver = tf.train.Saver(var_list = net.variables ,max_to_keep=40)


    ############DDPG#####################

    a_dim=NUM_CLASS;s_dim=NUM_CLASS*4;a_bound=10;

    ddpg,var = DDPG.DDPG_Prepare(a_dim,s_dim,a_bound)



  	###########Seg_Net & Q_Seg_Net ####################


    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # summary

    merged_train = tf.summary.merge_all('train') 
    merged_val = tf.summary.merge_all('val')


    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    summary_writer = tf.summary.FileWriter(args.log_dir,graph = tf.get_default_graph())


    # restore finetune

    if args.restore_from is not None:
        load(saver, sess, args.restore_from)
    

    # Start queue threads.

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    is_updata_q_net_to_net = False
    is_updata_net_to_q_net = True

    duration = 0

    distribution = np.zeros(NUM_CLASS)
    distribution_batch = np.zeros(NUM_CLASS)
    average_count = 0
    DDPG_count = 0


    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        if step % args.save_pred_every == 0 and step != 0:

            print("q_net >>>> net",is_updata_q_net_to_net)
            if is_updata_q_net_to_net:
                sess.run(updata_q_net_to_net_ops)

            loss_value, images, labels, preds, _, train_metrics_value,_ = sess.run([loss,image_batch, label_batch, pred, optim,train_metrics,train_updata_op],\
            														feed_dict={is_training: True})
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0]))

            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)

            print('step {:<6d}, loss = {:.5f}, miou = {:.5f}, ({:.5f} sec/step)'.format(step,loss_value,\
                               train_metrics_value,duration))



        else:
        	#################summary sumary

            print("q_net >>>> net",is_updata_q_net_to_net)

            if is_updata_q_net_to_net:
                sess.run(updata_q_net_to_net_ops)

            # _ ,loss_value,train_metrics_value,featuremap_val,image_batch_fetch,label_batch_fetch= \
            #                         sess.run([optim,loss,train_metrics,featuremap,image_batch,label_batch],\
            #                                 feed_dict={is_training:True})



            summary_train,loss_value, _ ,train_metrics_value,_,image_batch_fetch,label_batch_fetch= \
                        sess.run([merged_train,loss,optim,train_metrics,train_updata_op,image_batch,label_batch],\
                                feed_dict={is_training:True})

            summary_writer.add_summary(summary_train,step)


            newest_loss_value,featuremap_val,label_equal_val = sess.run([newest_loss,featuremap_get,label_equal],feed_dict={loss_image_batch:image_batch_fetch,loss_label_batch:label_batch_fetch})

            # print (loss_value,newest_loss_value)


            # loss_value,train_metrics_value,featuremap_val,image_batch_fetch,label_batch_fetch= \
            #                         sess.run([loss,train_metrics,featuremap,image_batch,label_batch],\
            #                                 feed_dict={is_training:True})

            # loss_origin,featuremap,image_batch,label_batch

            print('step {:<6d}, loss = {:.5f}, miou = {:.5f}, ({:.5f} sec/step)'.format(step,newest_loss_value,\
                                       train_metrics_value,duration))


            # #select = 0

            # s_0 = np.stack((np.argmax(label_equal_val[select],axis=2),np.argmax(featuremap_val[select],axis = 2)),axis = 2)

            # action_set = [1.0,0.1,100]


            label_flatten = np.argmax(label_equal_val,axis =3).flatten()
            featuremap_flatten = np.argmax(featuremap_val,axis =3).flatten()

            average_count += 1
            s_0,distribution,distribution_batch = calculate_state(label_flatten,featuremap_flatten,average_count,distribution,distribution_batch,False)




            def seg_get_state(a_t):

                w1 = a_t
                print("a",a_t)

                ##backup update Q_net parameters
                Q_loss_w_val,Q_featuremap_w_val,_ = sess.run([Q_loss_w,Q_featuremap_w,Q_optim],feed_dict={Q_image_batch:image_batch_fetch,Q_label_batch:label_batch_fetch,Q_action:w1})
                
                ##forward get Q_net feature and newestlosss
                Q_loss_val,Q_featuremap_val = sess.run([Q_loss,Q_featuremap],feed_dict={Q_image_batch:image_batch_fetch,Q_label_batch:label_batch_fetch})


                r_t = newest_loss_value - Q_loss_val

                print("orgin_loss:%.5f    q_loss_w:%.5f     q_loss:%.5f  r_t:%5f "%(loss_value, Q_loss_w_val, Q_loss_val,r_t))


                Q_label_flatten = np.argmax(label_equal_val,axis=3).flatten()
                Q_featuremap_val_argmax_flatten = np.argmax(Q_featuremap_val,axis=3).flatten()

                s_t1,_,_ = calculate_state(Q_label_flatten,Q_featuremap_val_argmax_flatten,average_count,distribution,distribution_batch,True)

                terminal = False

                return s_t1,r_t,terminal


            is_updata_net_to_q_net = True

            sess.run(updata_net_to_q_net_ops)

            is_updata_q_net_to_net,DDPG_count,var = DDPG.trainNetwork(ddpg,s_0,seg_get_state,DDPG_count,var)

            #is_updata_q_net_to_net =False



        duration = time.time() - start_time

        if step%1000 == 0 and step != 0:
            for i in range(VAL_LOOP):
                print(VAL_LOOP)
                start_time = time.time()
                summary_val,images, labels, preds,val_metrics_value,_ = sess.run([merged_val,image_batch, label_batch, pred,val_metrics,val_updata_op],feed_dict={is_training:False})
                
                summary_writer.add_summary(summary_val,step)

                for j in range(BATCH_SIZE):

                    fig, axes = plt.subplots(1, 3, figsize = (16, 12))

                    axes.flat[0].set_title('data')
                    axes.flat[0].imshow((images[j] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                    axes.flat[1].set_title('mask')
                    axes.flat[1].imshow(decode_labels(labels[j, :, :, 0]))

                    axes.flat[2].set_title('pred')
                    axes.flat[2].imshow(decode_labels(preds[j, :, :, 0]))

                    plt.savefig(args.save_dir + str(start_time) +'_'+str(i*BATCH_SIZE+j)+"test.png")
                    plt.close(fig)
                
                duration = time.time() - start_time
                
                print('step {:<6d}, val:  miou = {:.5f}, ({:.5f} sec/step)'.format(step, \
                            val_metrics_value,duration))


            save(saver, sess, args.snapshot_dir, step)

        # print('step {:<6d} \t loss = {:.5f}, precision = {:.5f}, recall = {:.5f}, accuracy = {:.5f}, ({:.5f} sec/step)'.format(step, loss_value,metrics[0],metrics[1],metrics[2],duration))
    
    summary_writer.close()

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
