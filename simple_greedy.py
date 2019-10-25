import numpy as np
import matplotlib.pyplot as plt

user_num = 5
k1 = 2
k2 = 1


Gd = np.random.triangular(0, 1, 4, [user_num, k1])
Gm = np.random.uniform(0, 2, user_num)
Gl = np.random.triangular(0, 5, 6, [user_num, k2])

Bm = 1  # macrocell 的 bandwidth
Bd = 1  # D2D cluster bandwidth
Bl = 1  # small cell bandwidth

# Pm = 1  # macrowave 的 power
# Pl = np.ones(user_num)   #D2D 的 power
# Pd = np.ones(user_num)   #small cell 的 power
Pm = 1
Pd = np.random.uniform(2, 4, k1)
Pl = np.random.uniform(2, 4, k2)
reward_his = []
sigma = 1


def cal_reward(index, allstate):
    max_index = 0
    reward_max = cal_state_reward(index, 0, allstate)
    for j in range(1+k1+k2):
        if cal_state_reward(index, j, allstate) > reward_max:
            max_index = j
            reward_max = cal_state_reward(index, j, allstate)
    allstate.append(max_index)
    return reward_max, allstate


def cal_state_reward(index, state_index, current_state):
    [M, D, L] = check_state(current_state)
    r_1 = 0
    r_2 = 0
    r_3 = 0
    # print('用户第', index, '个')
    # print('状态为', state_index, '个')
    if state_index == 0:
        M.append(index)
        r_1 = Bm * np.log2(Gm[index] * Pm / sigma + 1)
        # r_1 = Bm * np.log2(np.min(Gm[M]) * Pm / sigma + 1)
    elif 0 < state_index <= k1:
        sum_d = 0
        # D.append(index)
        for i in range(k1):
            sum_d += Gd[index, i]
        r_2 = Bd * np.log2(1 + Gd[index, state_index-1] / (sum_d - Gd[index, state_index-1] + sigma))
        # D.append(index)
        # sum_d = 0
        # for i in range(len(D)):
        #     sum_d += Gd[index, D[i]] * Pd[D[i]]
        # r_2 = Bd * np.log2(1 + (Gd[D[-1], state_index-1] * Pd[D[-1]]
        # / (sum_d - Gd[D[-1], state_index-1] * Pd[D[-1]] + sigma)))
    else:
        # L.append(index)
        sum_l = 0
        for i in range(k2):
            sum_l += Gl[index, i]
        r_3 = Bl * np.log2(1 + Gl[index, state_index - 1 - k1] / (sum_l - Gl[index, state_index - 1 - k1] + sigma))
    # print('奖励为', r_1+r_2+r_3)
    return r_1+r_2+r_3


def check_state(current_state):
    MM = []
    DD = []
    LL = []
    for i in range(len(current_state)):
        if current_state[i] == 0:
            MM.append(i)
        elif 0 < current_state[i] <= k1:
            DD.append(i)
        else:
            LL.append(i)
    return MM, DD, LL


def cal_reward_sum(actions):

    M = []
    D = []
    L = []
    reward = []
    # k1 , k2

    user_num = len(actions)
        # # D2D cell 用户的信道增益 (假设的是所有的用户的D2D信道增益，
        # # 在其中可能有一些用户并没有选择D2D方式)
        # Gd = np.ones(user_num)
        #
        # # Macrocell 用户的信道增益(假设的是所有用户的macrocell 信道增益，
        # # 在其中可能有一些用户并没有选择macrocell方式)
        # Gm = np.ones(user_num)
        #
        # # small cell 用户的信道增益(假设的是所有用户的small cell 信道增益，
        # # 在其中可能有一些用户并没有选择small cell方式)
        # Gl = np.ones(user_num)

    for i in range(len(actions)):
        if actions[i] == 0:
            M.append(i)
        elif 0 < actions[i] <= k1:
            D.append(i)
        else:
            L.append(i)

    for i in range(len(actions)):
        r = 0
        if actions[i] == 0:
            r = Bm * np.log2(np.min(Gm[M]) * Pm / sigma + 1)
        elif 0 < actions[i] <= k1:
            sum_d = 0
            # D.append(index)
            for j in range(k1):
                sum_d += Gd[i, j]
            r = Bd * np.log2(1 + Gd[i, actions[i] - 1] / (sum_d - Gd[i, actions[i] - 1] + sigma))
        else:
            sum_l = 0
            # D.append(index)
            for j in range(k2):
                sum_l += Gl[i, j]
            # print(actions[i])
            r = Bl * np.log2(1 + Gl[i, actions[i] - 1 - k1] / (sum_l - Gl[i, actions[i] - 1 - k1] + sigma))
        reward.append(r)
    return reward


# print('Pd', Pd)
# print('Pl', Pl)
# print('Gd', Gd)
# print('Gl', Gl)
# print('Gm', Gm)
# reward_his_sum = []
# episodes = 5500
# for episode in range(episodes):
#     print(episode)
#     state = []
#     reward = 0
#     np.random.shuffle(Gl)
#     np.random.shuffle(Gm)
#     np.random.shuffle(Pd)
#     np.random.shuffle(Gd)
#     np.random.shuffle(Pl)
#     # users = np.arange(user_num)
#     # np.random.shuffle(users)
#     # print(users)
#     # for i in range(len(users)):
#     for i in range(user_num):
#         reward, state = cal_reward(i, state)
#     # print(state)
#     reward_r = cal_reward_sum(state)
#     reward_sum = 0
#     for i in range(len(reward_r)):
#         # print(len(reward_r))
#         reward_sum += reward_r[i]
#     print(reward_sum)
#     reward_his_sum.append(reward_sum)

# np.save("reward_his_sum.npy", reward_his_sum)\
reward_sum = np.load("reward_sum.npy")
reward_his_sum = np.load("reward_his_sum.npy")
dqn_max = np.load("DQN_MAX.npy")
reward_sum_modified = []
# reward_sum = np.load("reward_sum.npy")
for i in range(len(reward_sum)):
    print(i % 700)
    if i % 700 == 0:
        reward_sum_modified.append(reward_sum[i])
dqn_array = [dqn_max for i in range(len(reward_sum_modified))]
plt.plot(np.arange(len(reward_his_sum)), reward_his_sum, label="Simple greedy")
plt.plot(np.arange(len(reward_sum_modified)), reward_sum_modified, label="Wolf PHC" )
plt.plot(np.arange(len(dqn_array)), dqn_array, label="DQN")
plt.xlabel("episode")
plt.ylabel("total user performance")
plt.xlim(0, 8)
# plt.ylim(10, 12)
plt.legend()
plt.savefig("result1.png")
plt.show()