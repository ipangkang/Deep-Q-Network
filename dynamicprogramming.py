import numpy as np

# 考虑到在该问题中，所有的用户的先后选择次序对结果并不会产生影响
# 那最佳策略等于 max{R(1) + max(N-pi(1)), R(2)+max(N-pi(2), R(3)+max(N-pi(3)))}

# 从最简单的一个用户开始考虑

user_num = 5
k1 = 2
k2 = 1

# 奖励可变怎么办？？？？
# 每次传进去的数应该是 state数组与user_num的结合在这里state应该刻画的是每一个用户采取的状态序列
def iteration(num, state, i):
    if i >= num:
        return 0, i
    else:
        array = []
        index = 0
        max = iteration(user_num, state.append(0), i+1) + cal_reward(0, i)
        for i in range(1 + k1 + k2):
            reward, j = iteration(user_num, state.append(i), i+1) + cal_reward(i, state)
            if reward > max:
                max = reward
                index = j
        state.append(index)
        return max, index


def cal_reward(state, index):
    return 0

iteration(user_num, [], 0)
# iteration(user_num, 2)
