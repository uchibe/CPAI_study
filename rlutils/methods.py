"""Bandit learning methods and experiment runners."""

import numpy as np
import matplotlib.pyplot as plt

from .policies import epsilon_greedy_policy, select_ucb_action


# ------------------------------------------------------------------
# uniform random baseline
# ------------------------------------------------------------------
def uniform_random_method(env, number_of_steps=1000, seed=None,
                          show_plot=True):
    """Single-run bandit experiment using a uniform random policy."""
    if show_plot:
        env.report_settings()
    rng = np.random.default_rng(None if seed is None else seed + 10_000)
    n_actions = env.n_actions

    Q_traj = np.zeros((n_actions, number_of_steps + 1))
    N_traj = np.zeros((n_actions, number_of_steps + 1))
    sum_r = np.zeros(n_actions)
    N_cnt = np.zeros(n_actions)
    rewards = np.zeros(number_of_steps)
    actions = np.zeros(number_of_steps, dtype=int)

    for t in range(number_of_steps):
        action = int(rng.choice(n_actions))
        reward = env.step(action)
        N_cnt[action] += 1
        sum_r[action] += reward
        N_traj[:, t + 1] = N_cnt
        for i in range(n_actions):
            Q_traj[i, t + 1] = (sum_r[i] / N_cnt[i]) if N_cnt[i] > 0 else 0.0
        rewards[t] = reward
        actions[t] = action

    if show_plot:
        _plot_run(Q_traj, N_traj, n_actions, "Uniform Random")
        print("final Q:", Q_traj[:, -1])
        print("average reward: %f" % rewards.mean())

    return dict(Q=Q_traj, N=N_traj, rewards=rewards, actions=actions,
                optimal_action=env.optimal_action)


# ------------------------------------------------------------------
# value-based (epsilon-greedy)
# ------------------------------------------------------------------
def value_based_method(env, epsilon=0.1, number_of_steps=1000,
                       initial_Q=0.0, seed=None, show_plot=True):
    """Single-run bandit experiment with epsilon-greedy action selection."""
    rng = np.random.default_rng(seed)
    n_actions = env.n_actions

    Q = np.zeros((n_actions, number_of_steps + 1))
    N = np.zeros((n_actions, number_of_steps + 1))
    Q[:, 0] = initial_Q
    rewards = np.zeros(number_of_steps)
    actions = np.zeros(number_of_steps, dtype=int)

    for t in range(number_of_steps):
        policy = epsilon_greedy_policy(Q[:, t], epsilon=epsilon)
        action = int(rng.choice(n_actions, p=policy))
        reward = env.step(action)

        N[:, t + 1] = N[:, t]
        N[action, t + 1] += 1
        Q[:, t + 1] = Q[:, t]
        alpha = 1.0 / N[action, t + 1]
        Q[action, t + 1] = Q[action, t] + alpha * (reward - Q[action, t])

        rewards[t] = reward
        actions[t] = action

    if show_plot:
        _plot_run(Q, N, n_actions, "Q-values over steps")
        print("final Q:", Q[:, -1])
        print("average reward: %f" % rewards.mean())

    return dict(Q=Q, N=N, rewards=rewards, actions=actions,
                optimal_action=env.optimal_action)


# ------------------------------------------------------------------
# UCB
# ------------------------------------------------------------------
def ucb_method(env, c=2.0, number_of_steps=1000,
               initial_Q=0.0, seed=None, show_plot=True):
    """Single-run bandit experiment using the UCB policy."""
    rng = np.random.default_rng(seed)
    n_actions = env.n_actions

    Q = np.zeros((n_actions, number_of_steps + 1))
    N = np.zeros((n_actions, number_of_steps + 1))
    Q[:, 0] = initial_Q
    rewards = np.zeros(number_of_steps)
    actions = np.zeros(number_of_steps, dtype=int)

    for t in range(number_of_steps):
        action = select_ucb_action(Q[:, t], N[:, t], t + 1, c=c, rng=rng)
        reward = env.step(action)

        N[:, t + 1] = N[:, t]
        N[action, t + 1] += 1
        Q[:, t + 1] = Q[:, t]
        alpha = 1.0 / N[action, t + 1]
        Q[action, t + 1] = Q[action, t] + alpha * (reward - Q[action, t])

        rewards[t] = reward
        actions[t] = action

    if show_plot:
        _plot_run(Q, N, n_actions, f"Q-values over steps (UCB c={c})")
        print("final Q:", Q[:, -1])
        print("average reward: %f" % rewards.mean())

    return dict(Q=Q, N=N, rewards=rewards, actions=actions,
                optimal_action=env.optimal_action)


# ------------------------------------------------------------------
# multi-run experiment runner
# ------------------------------------------------------------------
def run_experiment(env, method_function, n_runs=200,
                   number_of_steps=1000, base_seed=0, **kwargs):
    """Run *n_runs* independent experiments on the same environment.

    Returns
    -------
    avg_reward : (T,) array
    opt_rate   : (T,) array
    Q_last     : (n_actions, T+1) array  -- last run's Q trajectory
    """
    avg_reward_sum = np.zeros(number_of_steps)
    opt_action_sum = np.zeros(number_of_steps)
    last = None

    for k in range(n_runs):
        result = method_function(
            env=env,
            number_of_steps=number_of_steps,
            seed=base_seed + k + 10_000,
            show_plot=False,
            **kwargs,
        )
        avg_reward_sum += result["rewards"]
        opt_action_sum += (
            result["actions"] == result["optimal_action"]
        ).astype(float)
        last = result

    return (avg_reward_sum / n_runs,
            opt_action_sum / n_runs,
            last["Q"])


# ------------------------------------------------------------------
# internal plot helper
# ------------------------------------------------------------------
def _plot_run(Q, N, n_actions, title=""):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(n_actions):
        axarr[0].plot(Q[i, :], label=f"$a{i}$")
    axarr[0].legend()
    axarr[0].set_xlabel("steps")
    axarr[0].set_ylabel("action value Q(a)")
    axarr[0].set_title(title)
    axarr[0].grid()
    for i in range(n_actions):
        axarr[1].plot(N[i, :], label=f"$N{i}$")
    axarr[1].legend()
    axarr[1].set_xlabel("steps")
    axarr[1].set_ylabel("N(a)")
    axarr[1].grid()
    plt.show()
