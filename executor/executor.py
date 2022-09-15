import numpy as np
import pandas as pd

from MDP.MDP import MDP
from configs.config import cfg
from enums.enums import *


class Executor(object):
    def __init__(self):
        """
        Class for running policy evaluation algorithms with different params several times and analysing results.
        """
        self.cfg = cfg
        np.random.seed(0)
        self.seeds = np.random.choice(int(5e5), self.cfg.runs_num, replace=False)
        self.map_names = [m.name for m in MapName]
        self.policy_types = [p.name for p in PolicyType]
        self.action_sets = [a.name for a in ActionSet]
        self.direct_solution_avg_time = {m: [] for m in self.map_names}
        self.iterative_solution_avg_time = {m: [] for m in self.map_names}
        self.mdp = MDP(cfg)

    def write_results_to_csv(self, map_name, action_set, policy_iteration_mean_time, value_iteration_mean_time,
                             policy_iteration_mean_iter, value_iteration_mean_iter):
        """
        Writes all experiments results to csv file.
        :param map_name: list of map sizes
        :param action_set: list of action_set (default or slippery)
        :param policy_iteration_mean_time: list of mean time per experiment (policy_iteration, run 200 times)
        :param value_iteration_mean_time: list of mean time per experiment (value_iteration, run 200 times)
        :param policy_iteration_mean_iter: list of mean iterations num per experiment (policy_iteration, run 200 times)
        :param value_iteration_mean_iter: list of mean iterations num per experiment (value_iteration, run 200 times)
        """
        df = pd.DataFrame()
        df['map_name'] = map_name
        df['action_set'] = action_set
        df['policy_iteration_mean_time'] = policy_iteration_mean_time
        df['value_iteration_mean_time'] = value_iteration_mean_time
        df['policy_iteration_mean_iter'] = policy_iteration_mean_iter
        df['value_iteration_mean_iter'] = value_iteration_mean_iter
        df.to_csv(self.cfg.results_file_path)
        print(f'Saved csv file with results to {self.cfg.results_file_path}.')
        self.mdp.visualization.plot_scatters()

    def run_sequence_of_experiments(self):
        """
        Runs sequence of experiments with different params.
        """
        action_set_list, map_name_list, policy_iteration, value_iteration = [], [], [], []
        policy_iteration_mean_time, value_iteration_mean_time = [], []
        policy_iteration_mean_iter, value_iteration_mean_iter = [], []

        for m_id, map_name in enumerate(self.map_names):
            self.cfg.map_name = map_name

            for a_id, action_set in enumerate(self.action_sets):
                self.cfg.action_set = action_set

                policy_iteration_time, value_iteration_time = [], []
                policy_iteration_iter, value_iteration_iter = [], []
                for run_id in range(self.cfg.runs_num):
                    np.random.seed(self.seeds[run_id])
                    self.mdp = MDP(self.cfg)
                    self.mdp.run()

                    policy_iteration_time.append(self.mdp.policy_iteration_time)
                    value_iteration_time.append(self.mdp.value_iteration_time)

                    policy_iteration_iter.append(self.mdp.policy_iteration_iter)
                    value_iteration_iter.append(self.mdp.value_iteration_iter)

                policy_iteration_mean_time.append(np.mean(policy_iteration_time))
                value_iteration_mean_time.append(np.mean(value_iteration_time))

                policy_iteration_mean_iter.append(np.mean(policy_iteration_iter))
                value_iteration_mean_iter.append(np.mean(value_iteration_iter))

                action_set_list.append(self.cfg.action_set)
                map_name_list.append(self.cfg.map_name)

        self.write_results_to_csv(map_name_list, action_set_list, policy_iteration_mean_time, value_iteration_mean_time,
                                  policy_iteration_mean_iter, value_iteration_mean_iter)

    def run(self):
        """
        Runs whole pipeline.
        """
        if self.cfg.run_single_exp:
            self.mdp.run()
            self.mdp.visualization.plot_scatters()
        else:
            self.run_sequence_of_experiments()


if __name__ == '__main__':
    executor = Executor()
    executor.run()
