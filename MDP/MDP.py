import gym
import numpy as np
import time

from utils.visualization import Visualization
from utils.data_utils import read_file


class MDP(object):
    """
    Class with policy iteration implementation.
    """

    def __init__(self, cfg):
        """
        Initializes class.
        :param cfg: config
        """
        self.cfg = cfg
        self.visualization = Visualization(cfg)

        self.init_env()
        self.show_env_info()

        self.transition_matrix = self.env.transition_matrix
        self.states_num = len(self.transition_matrix)
        self.actions_num = len(self.transition_matrix[0])
        self.actions_space = self.env.action_space.n
        self.init_matrices()

    def init_env(self):
        """
        Initializes environment with parameters from config.
        """
        self.env = gym.make('frozen_lake:default-v0', map_name=self.cfg.map_name, action_set_name=self.cfg.action_set)
        self.env.reset(start_state_index=0)

    def show_env_info(self):
        """
        Prints information about current environment.
        """
        if self.cfg.verbose:
            self.env.render(object_type="environment")
            self.env.render(object_type="actions")
            self.env.render(object_type="states")
            print(f'\nMap type: {self.cfg.map_name}\n'
                  f'Policy type: {self.cfg.policy_type}\n'
                  f'Action Set: {self.cfg.action_set}\n'
                  f'Discount Factor: {self.cfg.discount_factor}')

    def init_matrices(self):
        """
        init transition probability matrix, reward matrix.
        """
        self.transition_prob_matrix = np.zeros((self.states_num, self.states_num, self.actions_num))
        self.reward_matrix = np.zeros((self.states_num, self.actions_num))

        for s in range(self.states_num):
            for a in range(self.actions_num):
                for new_s_tuple in self.transition_matrix[s][a]:
                    transition_prob, new_s, reward, _ = new_s_tuple
                    self.transition_prob_matrix[s][new_s][a] += transition_prob
                    self.reward_matrix[s][a] += reward * transition_prob

    def get_policy(self):
        """
        Initializes policy within config params.
        """
        if self.cfg.policy_type == 'stochastic':
            self.policy = np.random.uniform(0, 1, (self.states_num, self.actions_num))
            self.policy = [self.policy[i, :] / np.sum(self.policy, 1)[i] for i in range(self.states_num)]

        elif self.cfg.policy_type == 'optimal':
            data = read_file(self.cfg.optimal_policies_file_path)
            self.policy = np.asarray(data[self.cfg.map_name][self.cfg.action_set])

        else:
            raise Exception

    def get_model_with_policy(self):
        """
        Gets transition probability matrix, reward matrix within chosen policy.
        :return:
        """
        self.transition_prob_matrix_pi = np.zeros((self.states_num, self.states_num))
        self.reward_matrix_pi = np.zeros(self.states_num)

        for s in range(self.states_num):
            for new_s in range(self.states_num):
                self.transition_prob_matrix_pi[s][new_s] = self.policy[s] @ self.transition_prob_matrix[s][new_s]

            self.reward_matrix_pi[s] = self.policy[s] @ self.reward_matrix[s]

    def get_v_pi_by_iterative_solution(self):
        """
        Gets iterative solution.
        :return: v vector
        """
        start_time = time.time()

        v = np.zeros(self.states_num)
        step = 0

        while True:
            dif = 0
            for s in range(self.states_num):
                prev_v = v[s]
                v[s] = self.reward_matrix_pi[s] + self.cfg.discount_factor * (self.transition_prob_matrix_pi[s] @ v)
                dif = np.max([dif, abs(prev_v - v[s])])
            step += 1

            if dif < self.cfg.estimation_accuracy_thr:
                break

        self.iterative_solution_time = time.time() - start_time
        if self.cfg.verbose:
            print(f'v: {v}')
            print(f'Iterative solution time: {self.iterative_solution_time} s\n')
        return v

    def run_policy_evaluation(self):
        """
        Runs Policy Evaluation algorithm.
        """
        self.get_model_with_policy()
        self.v = self.get_v_pi_by_iterative_solution()

    def get_actions_values(self, state):
        """
        Gets actions values for given state.
        :param state: given state
        :return: actions_values list
        """
        actions_values = []
        for a in range(self.actions_num):
            a_value = self.reward_matrix[state][a] + self.cfg.discount_factor * \
                      (self.transition_prob_matrix[state, :, a] @ self.v)
            actions_values.append(a_value)
        return actions_values

    def get_optimal_action(self, state):
        """
        Gets optimal action based on actions_values list.
        :param state: given state
        :return: optimal action index
        """
        optimal_action = np.argmax(self.get_actions_values(state))
        return optimal_action

    def get_new_policy(self):
        """
        Gets new policy based on optimal actions.
        :return:
        """
        new_policy = np.zeros((self.states_num, self.actions_num))
        for s in range(self.states_num):
            optimal_action = self.get_optimal_action(s)
            new_policy[s][optimal_action] = 1
        return new_policy

    def run_policy_improvement(self):
        """
        Runs Policy Improvement step in Policy Iteration.
        :return:
        """
        old_policy = self.policy
        new_policy = self.get_new_policy()
        self.policy_stable = np.all(old_policy == new_policy)
        self.policy = new_policy

    def run_policy_iteration(self):
        """
        Policy evaluation and policy improvement loops.
        :return:
        """
        self.get_policy()
        # self.get_model_with_policy()

        self.policy_stable = False
        steps = 0
        while not self.policy_stable:
            print('Policy evaluation...')
            self.run_policy_evaluation()

            print('Policy improvement...')
            self.run_policy_improvement()

            steps += 1

            if not self.policy_stable:
                print('Policy is not stable, going back to policy evaluation...\n')
            else:
                print('Policy is stable.')

                self.show_v(self.v, alg_name='Policy Iteration')
                self.show_policy(self.policy, alg_name='Policy Iteration')

        return self.v, self.policy, steps

    def get_v_for_value_iteration(self):
        """
        Gets v for value_iteration algorithm.
        :return: v vector
        """
        start_time = time.time()

        self.v = np.zeros(self.states_num)
        step = 0

        while True:
            dif = 0
            for s in range(self.states_num):
                prev_v = self.v[s]
                actions_values = self.get_actions_values(s)
                self.v[s] = np.max(actions_values)
                dif = np.max([dif, abs(prev_v - self.v[s])])
            # print(f'step: {step}, dif: {dif}')
            step += 1

            if dif < self.cfg.estimation_accuracy_thr:
                break

        self.iterative_solution_time = time.time() - start_time
        if self.cfg.verbose:
            print(f'v: {self.v}')
            print(f'Iterative solution time: {self.iterative_solution_time} s\n')
        return step

    def run_value_iteration(self):
        """
        Runs Value Iteration algorithm.
        :return: v vector, optimal policy, number of iterations
        """
        steps_num = self.get_v_for_value_iteration()
        self.policy = self.get_new_policy()

        self.show_v(self.v, alg_name='Value Iteration')
        self.show_policy(self.policy, alg_name='Value Iteration')

        return self.v, self.policy, steps_num

    def show_policy(self, pi, alg_name=''):
        """
        Outputs Pi as matrix of size SxA and as action_space.
        """
        print(f'Policy (as SxA matrix):\n{pi}')
        if self.cfg.save_plots:
            self.visualization.plot_heatmap(file_name=f'Policy as SxA matrix, {alg_name}, '
                                                      f'action_set {self.cfg.action_set}',
                                            matrix=pi,
                                            title=f'Policy as SxA matrix, {alg_name}, '
                                                  f'action_set {self.cfg.action_set}',
                                            annotation_text=pi.astype('int'))

        pi = np.argmax(pi, -1).reshape((self.env.amount_rows, self.env.amount_columns))
        print(f'Policy (as action space):\n{pi}')

        if self.cfg.save_plots:
            self.visualization.plot_heatmap(file_name=f'Policy as action space, {alg_name}, '
                                                      f'action_set {self.cfg.action_set}',
                                            matrix=pi,
                                            title=f'Policy as action space, {alg_name}, '
                                                  f'action_set {self.cfg.action_set}',
                                            annotation_text=pi.astype('int'))

    def show_v(self, v, alg_name=''):
        """
        Outputs v as vector of size S and as action_space.
        """
        print(f'V as vector of size S:\n{v}')
        v = v.reshape((self.env.amount_rows, self.env.amount_columns))
        print(f'V as action space:\n{v}')

        if self.cfg.save_plots:
            self.visualization.plot_heatmap(file_name=f'V as action space, {alg_name}, '
                                                      f'action_set {self.cfg.action_set}',
                                            matrix=v,
                                            title=f'V as action space, {alg_name}, '
                                                  f'action_set {self.cfg.action_set}',
                                            annotation_text=np.round(v, 3))

    def run(self):
        """
        Runs Policy Iteration and Value Iteration algorithms.
        """
        policy_iteration_start = time.time()
        print('\nPolicy Iteration...\n')
        v_pi, policy_pi, steps_num_pi = self.run_policy_iteration()
        policy_iteration_end = time.time()

        value_iteration_start = time.time()
        print('\nValue Iteration...\n')
        v_vi, policy_vi, steps_num_vi = self.run_value_iteration()
        value_iteration_end = time.time()

        print(f'\nSteps num:\nPolicy Iteration - {steps_num_pi}\nValue Iteration - {steps_num_vi}')

        self.policy_iteration_iter = steps_num_pi
        self.value_iteration_iter = steps_num_vi

        self.policy_iteration_time = policy_iteration_end - policy_iteration_start
        self.value_iteration_time = value_iteration_end - value_iteration_start

        print(f'\nPolicy iteration time: {self.policy_iteration_time} sec')
        print(f'Value iteration time: {self.value_iteration_time} sec')
