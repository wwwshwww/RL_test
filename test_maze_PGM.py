import numpy as np

np.set_printoptions(precision=3 ,suppress=True)

def main():
    theta_0 = np.array([[np.nan, 1, 1, np.nan],
                        [np.nan, 1, np.nan, 1],
                        [np.nan, np.nan, 1, 1],
                        [1, 1, 1, np.nan],
                        [np.nan, np.nan, 1, 1],
                        [1, np.nan, np.nan, np.nan],
                        [1, np.nan, np.nan, np.nan],
                        [1, 1, np.nan, np.nan]])
    pi_0 = get_pi(theta_0)

    # pi = get_pi(theta_0)
    # sa_his = goal_maze(pi)
    # theta_updated = update_theta(theta_0, pi, sa_his)
    # print(sa_his)
    # print(pi)
    # print(get_pi(theta_updated))

    stop_epsilon = 10**-5

    theta = theta_0
    pi = pi_0

    is_continue = True
    
    count = 1
    while is_continue:
        s_a_history = goal_maze(pi)
        new_theta = update_theta(theta, pi, s_a_history)
        new_pi = get_pi(new_theta)

        aoc = np.sum(np.abs(new_pi - pi))

        print(f'count: {count}')
        print(f'AoC: {aoc}')
        print(f'step: {len(s_a_history)} \n--')

        if aoc < stop_epsilon:
            is_continue = False
        else:
            theta = new_theta
            pi = new_pi

        count += 1
    
    print(pi_0, '\n', pi)
    print(count)

def get_pi(theta, beta=1.0):
    exp_theta = np.exp(beta*theta)
    return np.nan_to_num([l/np.nansum(l) for l in exp_theta])

def get_next_s(pi, s):
    dire = ['up', 'right', 'down', 'left']
    dire_next = np.random.choice(dire, p=pi[s, :])
    
    if dire_next == 'up':
        action = 0
        s_next = s-3
    elif dire_next == 'right':
        action = 1
        s_next = s+1
    elif dire_next == 'down':
        action = 2
        s_next = s+3
    elif dire_next == 'left':
        action = 3
        s_next = s-1

    return action, s_next

def goal_maze(pi):
    s = 0
    s_a_history = [[0, np.nan]]

    while True:
        action, s_next = get_next_s(pi, s)
        s_a_history[-1][1] = action

        s_a_history.append([s_next, np.nan])

        if s_next == 8:
            break
        else:
            s = s_next

    return s_a_history

def update_theta(theta, pi, sa_his, eta=1.0):
    T = len(sa_his) - 1

    delta_theta = theta.copy()

    n, m = theta.shape
    for i in range(n):
        for j in range(m):
            if not np.isnan(theta[i,j]):
                n_sa_ij = len(list(filter(lambda sa: sa == [i,j], sa_his)))
                n_sa_i = len(list(filter(lambda sa: sa[0] == i, sa_his)))

                delta_theta[i,j] = (n_sa_ij - pi[i,j] * n_sa_i) / T

    new_theta = theta + eta * delta_theta
    return new_theta

if __name__ == "__main__":
    main()