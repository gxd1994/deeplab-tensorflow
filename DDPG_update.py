"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import random,os
from collections import deque

# import gym

# np.random.seed(1)
# tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EP_STEPS = 1 #400
LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 7000 #7000
OBSERVE = 100

BATCH_SIZE = 32 #32

EXPLORE = 20000   #2000000.
FINAL_VAR = 0.00001

INITIAL_VAR = 3 

DDPG_model = './DDPG_model/'
SAVE_PRE = 5000

# RENDER = False
# ENV_NAME = 'Pendulum-v0'

###############################  Actor  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        # self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory = deque()
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None,1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        save_list = self.ae_params + self.at_params + self.ce_params + self.ct_params
        self.saver = tf.train.Saver(var_list=save_list)

        q_target = self.R + GAMMA * q_
        print_op1 = tf.Print(q_target,[q_target[0]],message='q_target')
        print_op2 = tf.Print(q,[q_target[0]],message='q')

        #with tf.control_dependencies([print_op1,print_op2]):
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        tf.initialize_variables
        #self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # hard replace parameters
        if self.a_replace_counter % REPLACE_ITER_A == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        if self.c_replace_counter % REPLACE_ITER_C == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.a_replace_counter += 1; self.c_replace_counter += 1

        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]

        bt = random.sample(self.memory,BATCH_SIZE)

        bs = [d[0] for d in bt] 
        ba = [d[1] for d in bt]
        br = [d[2] for d in bt]
        bs_ = [d[3] for d in bt]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.pointer += 1

        self.memory.append((s,a,[r],s_))
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            lay1 = tf.layers.dense(s, 60, activation=tf.nn.relu, name='l1', trainable=trainable)
            lay2 = tf.layers.dense(lay1, 30, activation=tf.nn.relu, name='l2', trainable=trainable)
            #net = tf.layers.dense(lay2, 30, activation=tf.nn.relu, name='l3', trainable=trainable)
            # net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)

            a = tf.layers.dense(lay2, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            lay1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            lay2 = tf.layers.dense(lay1, 30, activation=tf.nn.relu, trainable=trainable)  
            lay3 = tf.layers.dense(lay2, 30, activation=tf.nn.relu, trainable=trainable)  

            return tf.layers.dense(lay3, 1, trainable=trainable)  # Q(s,a)



def DDPG_Prepare(a_dim,s_dim,a_bound):

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = INITIAL_VAR

    return ddpg,var




def trainNetwork(ddpg,s_0,seg_get_state,step,var):


    is_updata_q_net_to_net = False

    s = s_0

    ep_reward = 0

    a_old = None
    step += 1
    for j in range(MAX_EP_STEPS):
        state = 'Explore'

        # if j==0:
        #     # Add exploration noise
        #     a = ddpg.choose_action(s)

        #     a = np.clip(np.random.normal(a, max(0,var)), 0, 10)    # add randomness to action selection for exploration

        #     a_old = a
        # else:
        #     a = a_old

        a = ddpg.choose_action(s)

        a = np.clip(np.random.normal(a, max(0,var)), 0, 10)    # add randomness to action selection for exploration

        s_, r, done = seg_get_state(a)

        #print("s,r,a,s_",s.shape,r.shape,a.shape,s_.shape)
        r_final = r * 100
        ddpg.store_transition(s, a, r_final , s_)


        if var > FINAL_VAR and len(ddpg.memory) >= OBSERVE:
            var -= (INITIAL_VAR - FINAL_VAR) / EXPLORE

        if len(ddpg.memory) >= OBSERVE:
            ddpg.learn()
            state = 'Exploit'

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            print('Episode:%5d'%step,'Episode_steps:%5d'%(step*MAX_EP_STEPS), ' Reward: %5f' % ep_reward, 'var: %.5f' % var,'state:%s'%state,'r_final:%.5f'%r_final)
            if r > 0:
                is_updata_q_net_to_net = True

        cur_step = (step-1)*MAX_EP_STEPS+j
        if cur_step % SAVE_PRE == 0 and cur_step != 0: 
            if not os.path.exists(DDPG_model):
                os.makedirs(DDPG_model)

            ddpg.saver.save(ddpg.sess,DDPG_model, global_step = cur_step )



    # if ep_reward > 0:
    #     is_updata_q_net_to_net = True


    return  is_updata_q_net_to_net,step,var