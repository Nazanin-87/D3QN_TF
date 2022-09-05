'''
    Shiva Kazemi
    8/3/2022
'''
import numpy as np
from Agent_D3QN import D3QNAgent
from Environment import Environ
import copy, json, argparse
from utils import plot_learning_curve
from scipy.io import *
import torch

if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_actions',  type=int, default=4, help='Number of actions')
    parser.add_argument('--Num', type=int, default=2, help='Number of users')
    parser.add_argument('--learningrate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--eps', type=float, default=0.9, help='epsilon')
    parser.add_argument('--Pmax',  type=float, default=0.01, help='maximum transmit power')
    parser.add_argument('--Noise', type=float, default=0.00000000000001, help='Noise')
    parser.add_argument('--negative_cost', type=float, default=-1.0, help='negative cost for panalty')
    parser.add_argument('--Rmin', type=float, default=1000000, help='Minimum QoS')
    parser.add_argument('--BW', type=float, default=180000, help='Bandwidth')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--nepisodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--nsteps', type=int, default=50, help='Number of steps')
    parser.add_argument('--layer1_size', type=int, default=256)
    parser.add_argument('--layer2_size', type=int, default=256)

    args = parser.parse_args()
    env = Environ(Num=args.Num, n_actions=args.n_actions, Pmax=args.Pmax, Noise=args.Noise, BW=args.BW, Rmin=args.Rmin, negative_cost=args.negative_cost )

    agent= D3QNAgent( lr=args.learningrate, gamma=args.gamma,
                                         n_actions=args.n_actions, epsilon=args.eps,
                                         batch_size=args.batch_size,
                                         input_shape=[args.Num])

    Total_EE=[]
    eps_history =[]
    episodes_positive  = 0

    for i in range(args.nepisodes):
        Loc = env.Location()
        h = env.PathGain(Loc)

        done = False

        observation = env.reset()
        Reward=0

        nstep=0

        while not done and nstep<=args.nsteps:
            nstep+=1

            action = agent.choose_action(observation)
            print('action: ', action)
            new_observation , reward , done , infor  = env.step(action, h )
            agent.store_transition(observation , new_observation , action, reward , int(done ))

            agent.learn()
            observation  = new_observation 

            Reward=reward


        if Reward>0 :
            episodes_positive +=1
            Total_EE.append(Reward )



    print('Max EE: ', max(Total_EE)/10**9)
    print('Avg EE: ', np.mean(Total_EE)/10**9)

    filename_1 = 'EE.png'
    x=[i+1 for i in range(episodes_positive )]
    plot_learning_curve(x, Total_EE, filename_1)


