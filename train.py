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

import deep_q_network as DQN



BATCH_SIZE = 6
DATA_DIRECTORY = './dataset/class3_def/' #'/home/VOCdevkit'

DATA_TRAIN_LIST_PATH = DATA_DIRECTORY+'/train.txt'
DATA_VAL_LIST_PATH = DATA_DIRECTORY+'/test.txt'

INPUT_SIZE = '505,505'
LEARNING_RATE = 0.0001
MEAN_IMG = tf.Variable(np.array((175,175,175)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000
RANDOM_SCALE = False  #True
RESTORE_FROM = None   #'./deeplab_lfov.ckpt'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = './util/net_weights.ckpt'
LOG_DIR = './log'

VAL_LOOP = int(math.ceil(float(29)/BATCH_SIZE))

IMG_MEAN = np.array((175,175,175), dtype=np.float32)

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
            coord)
        image_batch_train, label_batch_train = reader_train.dequeue(args.batch_size)

    with tf.name_scope("create_val_inputs"):
        reader_val = ImageReader(
            args.data_dir,
            args.data_val_list,
            input_size,
            False,
            coord)
        image_batch_val, label_batch_val = reader_val.dequeue(args.batch_size)

    is_training = tf.placeholder(tf.bool,shape = [],name = 'stauts')


    image_batch,label_batch = tf.cond(is_training,lambda: (image_batch_train,label_batch_train),lambda: (image_batch_val,label_batch_val))

    # print (image_batch,label_batch)

    Q_image_batch = tf.placeholder(tf.float32,[BATCH_SIZE,505,505,3])
    Q_label_batch = tf.placeholder(tf.uint8,[BATCH_SIZE,505,505,1])



    Q_action = tf.placeholder(tf.float32,shape=[])




    # Create network.

    net = DeepLabLFOVModel(args.weights_path)

    # Define the loss and optimisation parameters.

    Q_net = DeepLabLFOVModel(args.weights_path)

    updata_net_to_q_net = tf.placeholder(tf.bool,shape=[])

    updata_q_net_to_net = tf.placeholder(tf.bool,shape=[])




    net_variables_final = tf.cond(updata_q_net_to_net,lambda: updata_fun(net.variables,Q_net.variables),lambda: net.variables)

    Q_net_variables_final = tf.cond(updata_net_to_q_net,lambda: updata_fun(Q_net.variables,net.variables),lambda: Q_net.variables)


    Q_loss_w,Q_train_metrics_w = Q_net.loss_w(Q_image_batch, Q_label_batch, Q_net_variables_final, Q_action)

    Q_loss,Q_train_metrics,Q_featuremap = Q_net.loss(Q_image_batch, Q_label_batch, Q_net_variables_final)

    loss,train_metrics,featuremap = net.loss(image_batch, label_batch, net_variables_final)


    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    trainable = tf.trainable_variables()


    optim = optimiser.minimize(loss)   #, var_list=trainable)

    Q_optim = optimiser.minimize(Q_loss_w)   #, var_list=trainable)


    
    pred = net.preds(image_batch)

    val_metrics = net.metrics(pred,label_batch)

    # tf.summary.scalar('val_r1',val_metrics[0])
    # tf.summary.scalar('val_r2',val_metrics[1])
    # tf.summary.scalar('val_p1',val_metrics[2])
    # tf.summary.scalar('val_p2',val_metrics[3])
    # tf.summary.scalar('val_acc',val_metrics[4])



    merged = tf.summary.merge_all() 

    if os.path.exists(args.log_dir):
    	shutil.rmtree(args.log_dir)

    summary_writer = tf.summary.FileWriter(args.log_dir,graph = tf.get_default_graph())




    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False


    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=trainable, max_to_keep=40)

    ########DQN#####################

    state,readout,h_fc1 = DQN.createNetwork()

    sess_q = tf.Session(config=config)

    sess_q,s,a,y,readout,train_step_q,epsilon,t,D = DQN.q_net_prepare(sess_q,saver,state,readout)

    

  	###########Seg_Net####################

    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    



    if args.restore_from is not None:
        load(saver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    is_updata_q_net_to_net = False
    is_updata_net_to_q_net = True

    duration = 0
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        if step % args.save_pred_every == 0:

            print("q_net >>>> net",is_updata_q_net_to_net)
            loss_value, images, labels, preds, _, train_metrics_value = sess.run( [loss,image_batch, label_batch, pred, optim,train_metrics],\
            														feed_dict={is_training: True, updata_q_net_to_net: is_updata_q_net_to_net})
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

            print('step {:<6d}, loss = {:.5f}, r0 = {:.5f}, r1 = {:.5f}, p0 = {:.5f}, p1 = {:.5f}, acc = {:.5f}, ({:.5f} sec/step)'.format(step,loss_value,\
                               train_metrics_value[0],train_metrics_value[1],train_metrics_value[2],train_metrics_value[3],train_metrics_value[4],duration))



        else:
        	#################summary sumary

            print("q_net >>>> net",is_updata_q_net_to_net)

            loss_value, _ ,train_metrics_value,featuremap_val,image_batch_val,label_batch_val= \
                                    sess.run([loss,optim,train_metrics,featuremap,image_batch,label_batch],\
                                            feed_dict={is_training:True, updata_q_net_to_net: is_updata_q_net_to_net})

            # loss_origin,featuremap,image_batch,label_batch

            print('step {:<6d}, loss = {:.5f}, r0 = {:.5f}, r1 = {:.5f}, p0 = {:.5f}, p1 = {:.5f}, acc = {:.5f}, ({:.5f} sec/step)'.format(step,loss_value,\
                                       train_metrics_value[0],train_metrics_value[1],train_metrics_value[2],train_metrics_value[3],train_metrics_value[4],duration))
			
            select = 0

            s_0 = featuremap_val[select]

            action_set = [1.0,0.1,10]


            def seg_get_state(a_t):

                a_t = 0

                w1 = action = action_set[int(a_t)]

                is_updata_net_to_q_net = False

                sess.run(Q_optim,feed_dict={updata_net_to_q_net:is_updata_net_to_q_net,Q_image_batch:image_batch_val,Q_label_batch:label_batch_val,Q_action:w1})

                # is_updata_net_to_q_net = False

                Q_loss_val,Q_featuremap_val = sess.run([Q_loss,Q_featuremap],feed_dict={updata_net_to_q_net:is_updata_net_to_q_net,Q_image_batch:image_batch_val,Q_label_batch:label_batch_val})


                print("orgin_loss:%.5f  q_loss:%.5f"%(loss_value,Q_loss_val))

                r_t = loss_value - Q_loss_val

                s_t1 = Q_featuremap_val[select]


                terminal = False

                return s_t1,r_t,terminal


            is_updata_net_to_q_net = True

            Q_net_variables_final_val = sess.run(Q_net_variables_final,feed_dict={updata_net_to_q_net:is_updata_net_to_q_net})

            epsilon,t,is_updata_q_net_to_net = DQN.trainNetwork(sess_q,saver,s,a,y,readout,train_step_q,epsilon,t,D,s_0,seg_get_state)

            # summary_writer.add_summary(summary,step)


        duration = time.time() - start_time



        if step%500 == 0 and step != 0:
			for i in range(VAL_LOOP):
				print(VAL_LOOP)
				start_time = time.time()
				images, labels, preds,val_metrics_value = sess.run([image_batch, label_batch, pred,val_metrics],feed_dict={is_training:False, \
                                                                                                             updata_q_net_to_net: is_updata_q_net_to_net})
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

        		print('step {:<6d}, val:  r0 = {:.5f}, r1 = {:.5f}, p0 = {:.5f}, p1 = {:.5f}, acc = {:.5f}, ({:.5f} sec/step)'.format(step, \
        					         val_metrics_value[0],val_metrics_value[1],val_metrics_value[2],val_metrics_value[3],val_metrics_value[4],duration))


			save(saver, sess, args.snapshot_dir, step)

        # print('step {:<6d} \t loss = {:.5f}, precision = {:.5f}, recall = {:.5f}, accuracy = {:.5f}, ({:.5f} sec/step)'.format(step, loss_value,metrics[0],metrics[1],metrics[2],duration))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
