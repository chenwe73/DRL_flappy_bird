import numpy as np
import matplotlib.pyplot as plt
import time

import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
from qnetwork import qnetwork
from utils import Memory, update_target_graph


def trackS20():
    W = 20
    track = np.zeros((W, W), dtype='int')
    track[2:4,2:20] = 1
    track[10:20,0:18] = 1
    return [W, track]

def trackS10():
    W = 10
    track = np.zeros((W, W), dtype='int')
    track[2:4,2:W] = 1
    track[6:10,0:8] = 1
    return [W, track]

def flappyBird():
    W = 40
    dist = 3
    map = np.zeros((W, W), dtype='int')
    map[W-dist] = 1
    map[W-dist,7:10] = 0
    return [W, map]
    
def empty():
    W = 20
    map = np.zeros((W, W), dtype='int')
    return [W, map]
    
class Environment:
    [W, map] = trackS10()
    nActions = 4
    stateShape = 4
    velocityBound = [-5, 5]
    nVelocity = velocityBound[1] - velocityBound[0] + 1
    discount = 0.99
    startPosition = np.array([0, W-1])
    startVelocity = np.array([0, 0])
    endPosition = np.array([0, W-1])
    gravity = np.array([0, 0])
    nSteps = 400
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.stateReset()
        self.action = 0
        self.time = 0
        state = np.append(self.position, self.velocity)
        return state
        
    def stateReset(self):
        self.position = np.array([0, np.random.randint(self.W)]) # np.copy(self.startPosition) #
        self.velocity = np.copy(self.startVelocity)
        
    def isFinish(self):
        #result = (self.position[0] >= self.W)
        result = (self.position[0] >= self.W and self.position[1] < self.W and self.position[1] >= 0)
        #result = (np.array_equal(self.position,self.endPosition) and np.array_equal(self.velocity, np.array([0,0])))
        return result
        
    def isOutOfBound(self):
        result = self.position[0] < 0 or self.position[0] >= self.W or self.position[1] < 0 or self.position[1] >= self.W
        if (not result):
            result = self.map[tuple(self.position)]
        return result
    
    def acceleration(self, action):
        a = 1
        result = {
            0: np.array([ a,  0]),
            1: np.array([ 0,  a]),
            2: np.array([-a,  0]),
            3: np.array([ 0, -a]),
            }
        return result[action]
    
    def flap(self, action):
        result = np.array([ 0,  0])
        if (action):
            result = np.array([ 0,  4])
        return result
    
    def step(self, action):
        reward = -1
        done = False
        self.action = action
        self.position += self.velocity
        self.velocity += self.acceleration(action)
        self.velocity += self.gravity
        self.velocity = np.clip(self.velocity, self.velocityBound[0], self.velocityBound[1])
        self.time += 1
        if (self.isFinish()):
            done = True
            self.position = np.copy(self.endPosition)
            reward = 10
            #self.position[0] = 0
        elif (self.isOutOfBound()):
            self.stateReset()
            #self.position = np.clip(self.position, 0, self.W-1)
            #reward = -100
        if (self.time >= self.nSteps):
            done = True
        nextState = np.append(self.position, self.velocity)
        return [nextState, reward, done]
    
    def render(self, trace = []):
        mapDraw = np.copy(self.map)
        for i in trace:
            mapDraw[tuple(i)] = 2
        for i in range(self.W+2):
            print('-'),
        print('')
        for i in range(self.W):
            y = self.W-1 - i
            print('|'),
            for j in range(self.W):
                if (y == int(self.position[1]) and j == int(self.position[0])):
                    print('*'),
                elif (mapDraw[j,y] == 1):
                    print('#'),
                elif (mapDraw[j,y] == 2):
                    print('+'),
                else:
                    print(' '),
            print('|')
        for i in range(self.W+2):
            print('-'),
        print('')
        print "action: ", self.action, ", position: ", self.position, ", velocity: ", self.velocity
        time.sleep(0.1)
        
def qLearning(Q0=None, epsilon=0.1, isDraw=False):
    nEpisodes = 1000
    #nSteps = 400
    env = Environment()
    discount = env.discount
    nActions = env.nActions
    W = env.W
    space = (env.W, env.W, env.nVelocity, env.nVelocity, env.nActions)
    if (Q0 is None): Q = np.zeros(space)
    else: Q = Q0
    N = np.zeros(space)
    rewardHistory = np.zeros(nEpisodes)
    done = False
    env.render()
    
    for episode in range(nEpisodes):
        state = env.reset()
        trace = []
        if (isDraw): env.render(trace)
        while(True):
            # select action
            #epsilon = 0.1
            if (np.random.rand(1) < epsilon):
                action = np.random.randint(nActions)
            else:
                action = np.argmax(Q[tuple(state)])
            # go to next state
            [nextState, reward, done] = env.step(action)
            rewardHistory[episode] += reward # * discount**step
            trace.append([state[0], state[1]])
            # update
            space = np.append(state, action)
            N[tuple(space)] += 1
            deltaQ = (reward + discount * Q[tuple(nextState)].max() - Q[tuple(space)])
            learningRate = 0.9 # 1.0 / N[tuple(space)]
            Q[tuple(space)] += learningRate * deltaQ
            state = np.copy(nextState)
            
            if (isDraw): env.render(trace)
            if (done): break
        #print(rewardHistory[episode])
    return [Q, rewardHistory]

def randomWalk():
    env = Environment()
    rTotal = 0
    for t in range(1000):
        action = np.random.randint(env.nActions)
        action = 0
        [position, velocity, reward, done] = env.step(action)
        rTotal += reward
        print (action, env.position)
        env.render()
        time.sleep(0.1)
        if done:
            print(rTotal)
            break

def test():
    [Q, rewardHistory] = qLearning()
    '''
    np.set_printoptions(threshold=np.nan)
    for i in range(-5,5):
        print(np.argmax(Q[:,:,1,i],axis=-1))
    '''
    plt.xlabel('episode #')
    plt.ylabel('reward')
    plt.plot(rewardHistory)
    plt.savefig("plot")
    plt.show()
    
    [Q, rewardHistory] = qLearning(Q0=Q, epsilon=0, isDraw=True)



def DQN():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default='CartPole-v0')
    parser.add_argument("--action-size", type=int, default=2)
    parser.add_argument("--input-shape", type=list, default=[None, 4])
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epsilon-max", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.01)
    parser.add_argument("--epsilon-decay", type=float, default=.001)

    parser.add_argument("--discount-factor", type=float, default=.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)

    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    args = parser.parse_args()

    env = Environment()
    args.action_size = env.nActions
    args.input_shape = [None, env.stateShape]

    print args

    # Epsilon parameter
    epsilon = 0.1 # args.epsilon_max

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Time step
    time_step = 0.

    # Initialize the agent
    qnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='qnet')
    tnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='tnet')
    update_ops = update_target_graph('qnet', 'tnet')
    
    rewardHistory = np.zeros(args.epochs)
    env.render()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            total_reward = 0
            state = env.reset()
            while(True):
                #env.render()
                if np.random.rand() < epsilon:
                    action = np.random.randint(args.action_size)
                else:
                    action = qnet.act(sess, state)
                [next_state, reward, done] = env.step(action)
                total_reward += reward
                rewardHistory[epoch] += reward

                # Add to memory
                memory.add([state, action, reward, next_state, done])

                # Reduce epsilon
                time_step += 1.
                #epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay * time_step)

                # Training step
                batch = np.array(memory.sample(args.batch_size))
                qnet.train(sess, batch, args.discount_factor, tnet)

                # s <- s'
                state = np.copy(next_state)

                # Update target network
                if int(time_step) % args.target_update_freq == 0:
                    sess.run(update_ops)

                if done:
                    print 'epoch:', epoch, 'total_rewards:', total_reward
                    break
        '''
        np.set_printoptions(threshold=np.nan)
        for v in range(-5, 5):
            policy = np.zeros((env.W, env.W), dtype='int')
            for x in range(env.W):
                for y in range(env.W):
                    policy[x,y] = qnet.act(sess, np.array([x,y,1,v]))
            print(policy)
        '''
        plt.xlabel('episode #')
        plt.ylabel('reward')
        plt.plot(rewardHistory)
        plt.savefig("DQN")
        plt.show()
        
        for epoch in range(10):
            total_reward = 0
            state = env.reset()
            while(True):
                env.render()
                action = qnet.act(sess, state)
                [next_state, reward, done] = env.step(action)
                total_reward += reward
                rewardHistory[epoch] += reward

                # Reduce epsilon
                time_step += 1.
                # s <- s'
                state = np.copy(next_state)

                if done:
                    print 'epoch:', epoch, 'total_rewards:', total_reward
                    break
        
        

if __name__ == '__main__':
    DQN()
