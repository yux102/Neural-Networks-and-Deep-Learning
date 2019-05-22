import gym
import tensorflow as tf
import numpy as np
import random

class Memory():
    def __init__(self, n_states, memory_size=100, batch_size=32):
        '''initial the class

        args:
            - memory_size, set size of the memory. default is 100
            - n_states, set the number of states
            - batch_size, set size of one batch. default is 32
        '''
        self.memory_size = memory_size
        self.n_states = n_states
        self.batch_size = batch_size
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2 + 1 + 1))

    def store_transition(self, s, a, r, done, s_):
        '''store the transition into the memory.

        args:
            - s: current state, list with 4 elements
            - a: next action, list with 2 elements
            - r: get reward, scale
            - done: done or not, scale
            - s_: next state, list with 4 elements
        '''
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, a, [r, done], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def get_batch_memory(self):
        '''sample batch memory from all memory.

        return:
            - batch_memory, a np.array within batch size data
        '''
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        return batch_memory

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.95 # discount factor
INITIAL_EPSILON = 0.9 # starting value of epsilon
FINAL_EPSILON =  0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 250 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON # 0.9
STATE_DIM = env.observation_space.shape[0] # 4
ACTION_DIM = env.action_space.n # 2
tf.reset_default_graph()
# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM], name='state_in')
action_in = tf.placeholder("float", [None, ACTION_DIM], name='action_in') # for q_action cal
target_in = tf.placeholder("float", [None], name='target_in') # for loss cal

# TODO: Define Network Graph
LEARNING_RATE = 0.0025
HIDDEN_DIM_1 = 64

with tf.variable_scope('eval_net'):
    # c_names(collections_names) are the collections to store variables
    c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
    w_initializer = tf.random_normal_initializer(0., 0.15)
    b_initializer = tf.constant_initializer(0.)

    # first layer. collections is used later when assign to target net
    with tf.variable_scope('l1'):
        params_w1 = tf.get_variable('params_w1', [STATE_DIM, HIDDEN_DIM_1], initializer=w_initializer, collections=c_names)
        params_b1 = tf.get_variable('params_b1', [HIDDEN_DIM_1], initializer=b_initializer, collections=c_names)
        hidden_layer_1 = tf.nn.relu(tf.matmul(state_in, params_w1) + params_b1)

    # second layer. collections is used later when assign to target net
    with tf.variable_scope('l2'):
        params_w2 = tf.get_variable('params_w2', [HIDDEN_DIM_1, ACTION_DIM], initializer=w_initializer, collections=c_names)
        params_b2 = tf.get_variable('params_b2', [ACTION_DIM], initializer=b_initializer, collections=c_names)

        q_values = tf.matmul(hidden_layer_1, params_w2) + params_b2


# TODO: Network outputs
#q_values = # see above
#q_action = tf.reduce_sum(-tf.log(q_values) * action_in, reduction_indices = 1)
q_action = tf.reduce_sum(q_values * action_in, reduction_indices = 1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss) # RMSPropOptimizer or AdamOptimizer


# ------------------ build target_net ------------------
state_in_ = tf.placeholder("float", [None, STATE_DIM], name='state_in_')    # input
with tf.variable_scope('target_net'):
    # c_names(collections_names) are the collections to store variables
    c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

    # first layer. collections is used later when assign to target net
    with tf.variable_scope('l1'):
        params_w1 = tf.get_variable('params_w1', [STATE_DIM, HIDDEN_DIM_1], initializer=w_initializer, collections=c_names)
        params_b1 = tf.get_variable('params_b1', [HIDDEN_DIM_1], initializer=b_initializer, collections=c_names)
        hidden_layer_1 = tf.nn.relu(tf.matmul(state_in_, params_w1) + params_b1)

    # second layer. collections is used later when assign to target net
    with tf.variable_scope('l2'):
        params_w2 = tf.get_variable('params_w2', [HIDDEN_DIM_1, ACTION_DIM], initializer=w_initializer, collections=c_names)
        params_b2 = tf.get_variable('params_b2', [ACTION_DIM], initializer=b_initializer, collections=c_names)

        q_target_ouput = tf.matmul(hidden_layer_1, params_w2) + params_b2

t_params = tf.get_collection('target_net_params')
e_params = tf.get_collection('eval_net_params')
replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


'''
HIDDEN_DIM = 10
LEARNING_RATE = 0.01
w_initializer = tf.random_normal_initializer(0., 0.3)
b_initializer = tf.constant_initializer(0.1)
params_w1 = tf.get_variable("policy_parameters_w1",[STATE_DIM, HIDDEN_DIM], initializer=w_initializer)
params_b1 = tf.get_variable("policy_parameters_b1", [HIDDEN_DIM], initializer=b_initializer)
params_w2 = tf.get_variable("policy_parameters_w2",[HIDDEN_DIM, ACTION_DIM], initializer=w_initializer)
params_b2 = tf.get_variable("policy_parameters_b2", [ACTION_DIM], initializer=b_initializer)

hidden_layer = tf.nn.relu(tf.matmul(state_in, params_w1) + params_b1)

# TODO: Network outputs
q_values = tf.nn.softmax(tf.matmul(hidden_layer, params_w2) + params_b2)
#q_values = tf.matmul(hidden_layer, params_w2) + params_b2
#print ("shape q_values:", q_values.shape)

q_action = tf.reduce_sum((q_values * action_in), reduction_indices = 1)
#print ("shape q_action:", q_action.shape)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss) # RMSPropOptimizer
'''

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


memory = Memory(memory_size=100, n_states=STATE_DIM, batch_size=32)
learn_step_counter = 1
replace_target_iter = 50
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    #epsilon = epsilon - epsilon / EPSILON_DECAY_STEPS if epsilon > FINAL_EPSILON else FINAL_EPSILON

    # Move through env according to e-greedy policy
    for step in range(STEP):
        ## change the structure
        action = explore(state, epsilon) # [0 1] or [1 0]; shape = (2,)
        #print (action)
        next_state, reward, done, _ = env.step(np.argmax(action))

        ## save to memory
        memory.store_transition(state, action, reward, done, next_state)

        ## if satisfied with the condition, start training
        if not (episode == 0 and step < 100):
        #if rue:

            # check to replace target parameters
            if learn_step_counter % replace_target_iter == 0:
                session.run(replace_target_op)
                #print('\ntarget_params_replaced\n')

            # sample batch memory from all memory
            batch_memory = memory.get_batch_memory()

            q_next = session.run(q_target_ouput, feed_dict={state_in_: batch_memory[:, -STATE_DIM:]}) # [?, 2]

            batch_reward = batch_memory[:, STATE_DIM + 2]
            batch_done = batch_memory[:, -STATE_DIM - 1]
            q_target = np.zeros((batch_memory.shape[0])).astype(np.float32)
            for i in range(batch_memory.shape[0]):
                if batch_done[i]:
                    q_target[i] = batch_reward[i]
                else:
                    q_target[i] = batch_reward[i] + GAMMA * np.max(q_next[i])

            # train eval network
            session.run([optimizer], feed_dict={
                target_in: q_target,
                action_in: batch_memory[:, STATE_DIM: STATE_DIM + 2],
                state_in: batch_memory[:, :STATE_DIM]
            })

            learn_step_counter += 1

            #############################################################
            '''nextstate_q_values = q_values.eval(feed_dict={
                state_in: [next_state]
            })
            #print ("nextstate_q_values:", nextstate_q_values, "\tshape:",nextstate_q_values.shape)

            # TODO: Calculate the target q-value.
            # hint1: Bellman
            # hint2: consider if the episode has terminated
            target = reward + GAMMA * np.max(nextstate_q_values[0]) if not done else reward

            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: [target],
                action_in: [action],
                state_in: [state]
            })'''

        # Update
        state = next_state
        if done:
            break

        '''
        action = explore(state, epsilon) # [0 1] or [1 0]; shape = (2,)
        next_state, reward, done, _ = env.step(np.argmax(action))

        ###
        #print ("next_state:", next_state)
        #print ("reward:", reward)
        #print ("done:", done)
        ###

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })
        #print ("nextstate_q_values:", nextstate_q_values, "\tshape:",nextstate_q_values.shape)

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        target = reward + GAMMA * np.max(nextstate_q_values[0]) if not done else reward

        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })

        # Update
        state = next_state
        if done:
            break
        '''

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
