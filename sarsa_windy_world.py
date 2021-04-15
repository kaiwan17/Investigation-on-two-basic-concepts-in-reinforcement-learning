#######################################################################
# 重要声明：该代码并不是作者完全自编写，只是在刘建平老师博客
# https://www.cnblogs.com/pinard/p/9614290.html的基础上，#
#加入了discount=γ，原程序中默认为1，在此感谢刘建平老师以及Shangtong Zhang、Kenta Shimada的贡献
# 读者可以改变discount=γ=0.2或0.8，来观察输出的最优状态值函数和最优策略，将会发现明显不同
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
##https://www.cnblogs.com/pinard/p/9614290.html ##
## 强化学习（六）时序差分在线控制算法SARSA ##

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# world height，定义虚拟世界的高度
WORLD_HEIGHT = 7

# world width，定义虚拟世界的宽度
WORLD_WIDTH = 10

# wind strength for each column，定义虚拟世界每列的风速
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions，定义动作集中包括的动作
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# probability for exploration，定义探索率
EPSILON = 0.1

# Sarsa step size，定义误差加权比值
ALPHA = 0.5

# reward for each step，定义每步即时奖励
REWARD = -1.0

# discount=γ，定义折扣因子γ
discount = 0.2

# 定义起始点
START = [3, 0]
# 定义目标点
GOAL = [3, 7]
# 定义动作集
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


# 定义根据当前状态和动作确定的下一状态
def step(state, action):
    # 定义当前状态
    i, j = state
    # 定义根据当前执行动作的不同得到的下一状态
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

# play for an episode，定义采样得到的一条轨迹
def episode(q_value):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + discount*q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time



# 定义sarsa函数
def sarsa():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    episode_limit = 1000000

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('./sarsa.png')
    #用于显示采样轨迹数量和时间步
    #plt.close()

    # display the optimal policy，显示最优策略
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))
    print(q_value)


# 主执行程序
if __name__ == '__main__':
    sarsa()
    
