import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3 ,suppress=True)

def simple_convert_into_pi_from_theta(theta: np.ndarray):
    m,n = theta.shape
    pi = np.zeros_like(theta)
    for i in range(m):
        pi[i,:] = theta[i,:] / np.nansum(theta[i,:])
        
    pi = np.nan_to_num(pi)

    return pi

def get_actions(s, Q: np.ndarray, epsilon: float, pi_0):
    direction = ["up", "right", "down", "left"]

    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s,:])
    else:
        next_direction = direction[np.nanargmax(Q[s,:])]

    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3

    return action

def get_s_next(s, a: int, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]

    if next_direction == "up":
        s_next = s-3
    elif next_direction == "right":
        s_next = s+1
    elif next_direction == "down":
        s_next = s+3
    elif next_direction == "left":
        s_next = s-1

    return s_next

def sarsa(s: int, a: int, r, s_next: int, a_next: int, Q, eta, gamma):
    if s_next == 8:
        Q[s,a] = Q[s,a] + eta * (r - Q[s,a])
    else:
        Q[s,a] = Q[s,a] + eta * (r + gamma * Q[s_next,a_next] - Q[s,a])

    return Q

def goal_maze(Q, epsilon, eta, gamma, pi):
    s = 0 # start
    a = a_next = get_actions(s, Q, epsilon, pi)
    s_a_his = [[0, np.nan]]

    while True:
        a = a_next

        s_a_his[-1][1] = a
        s_next = get_s_next(s, a, Q, epsilon, pi)
        s_a_his.append([s_next, np.nan])

        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_actions(s_next, Q, epsilon, pi)

        Q = sarsa(s, a, r, s_next, a_next, Q, eta, gamma)

        if s_next == 8:
            break
        else:
            s = s_next

    return s_a_his, Q

def main():
    theta_0 = np.array([[np.nan, 1, 1, np.nan],
                        [np.nan, 1, np.nan, 1],
                        [np.nan, np.nan, 1, 1],
                        [1, 1, 1, np.nan],
                        [np.nan, np.nan, 1, 1],
                        [1, np.nan, np.nan, np.nan],
                        [1, np.nan, np.nan, np.nan],
                        [1, 1, np.nan, np.nan]])

    pi_0 = simple_convert_into_pi_from_theta(theta_0)

    a, b = theta_0.shape
    Q = np.random.rand(a,b) * theta_0


    eta = 0.1
    gamma = 0.9
    epsilon = 0.5
    v = np.argmax(Q, axis=1)
    is_continue = True
    episode = 1

    while True:
        print(f"episode: {episode}")

        epsilon /= 2

        s_a_his, Q = goal_maze(Q, epsilon, eta, gamma, pi_0)

        new_v = np.nanmax(Q, axis=1)
        print(f"td: {np.sum(np.abs(new_v - v))}")
        v = new_v

        print(f"solve steps: {len(s_a_his)-1}\n---------------")

        episode += 1
        if episode > 100:
            break

if __name__ == "__main__":
    main()