import numpy as np


user_num = 100
k1 = 2
k2 = 3

state = []
reward = 0
Bm = 1
Bd = 1
Bl = 1
Pm = 1
Pd = np.random.uniform(0, 1, k1)
Pl = np.random.uniform(2, 4, k2)
sigma = 1
Gd = np.random.uniform(0.5, 1.1, [user_num, k1])
Gm = np.random.uniform(0.5, 1, user_num)
Gl = np.random.uniform(40, 60, [user_num, k2])


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
        r_1 = Bm * np.log2(np.min(Gm[M]) * Pm / sigma + 1)
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
            r_2 = Bl * np.log2(1 + Gl[index, state_index - 1 - k1] / (sum_l - Gl[index, state_index - 1 - k1] + sigma))
    # print('奖励为', r_1+r_2+r_3)
    return r_1+r_2+r_3


def check_state(current_state):
    MM = []
    DD = []
    LL = []
    for i in range(len(current_state)):
        if current_state[i] == 0:
            MM.append(i)
        elif 0 < state[i] <= k1:
            DD.append(i)
        else:
            LL.append(i)
    return MM, DD, LL


# print('Pd', Pd)
# print('Pl', Pl)
# print('Gd', Gd)
# print('Gl', Gl)
print('Gm', Gm)
for i in range(user_num):
    reward, state = cal_reward(i, state)
    print(state)

