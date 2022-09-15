from easydict import EasyDict

from enums.enums import *

cfg = EasyDict()

cfg.results_file_path = '../data/results.csv'

cfg.discount_factor = 0.9
cfg.estimation_accuracy_thr = 1e-5
cfg.policy_type = PolicyType.stochastic.name
cfg.map_name = MapName.small.name

cfg.action_set = ActionSet.default.name  # изменить action set: ActionSet.default.name, ActionSet.slippery.name

cfg.verbose = True
cfg.runs_num = 200
cfg.run_single_exp = True
cfg.plots_dir = '../plots/'
cfg.save_plots = False
cfg.show_heatmap = False
