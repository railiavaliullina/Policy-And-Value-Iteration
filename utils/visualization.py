import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd


class Visualization(object):
    def __init__(self, cfg):
        """
        Class for visualizing policy, value as action set; run time and convergence time comparison plots.
        :param cfg: config
        """
        self.cfg = cfg

    def plot_heatmap(self, file_name, matrix=None, title='', annotation_text=None):
        """
        Saves heatmaps for policy and value as action set.
        :param file_name: filename to save plot with
        :param matrix: data for heatmap
        :param title: plot title
        :param annotation_text: annotation text for heatmap
        :return:
        """
        y_labels = list(np.arange(matrix.shape[0]).astype('str'))
        x_labels = list(np.arange(matrix.shape[1]).astype('str'))

        fig = ff.create_annotated_heatmap(matrix, x=x_labels, y=y_labels,
                                          annotation_text=annotation_text,
                                          showscale=True, colorscale='Viridis')

        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.color = 'white'
            # fig.layout.annotations[i].font.size = 16

        fig.update_layout(title=title)
        fig['layout']['yaxis']['autorange'] = "reversed"
        fig.update_xaxes(side="top")
        fig.write_image(self.cfg.plots_dir + f"{file_name}.png")

        if self.cfg.show_heatmap:
            fig.show()

    def plot_scatters(self):
        """
        Plots run time and convergence time comparison plots.
        Plots will be visualized in browser with ability to zoom in and will be saved as files.
        """
        df = pd.read_csv(self.cfg.results_file_path)
        map_name = df.map_name.to_list()
        action_set = df.action_set.to_list()
        policy_iteration_mean_time = df.policy_iteration_mean_time.to_list()
        value_iteration_mean_time = df.value_iteration_mean_time.to_list()
        policy_iteration_mean_iter = df.policy_iteration_mean_iter.to_list()
        value_iteration_mean_iter = df.value_iteration_mean_iter.to_list()

        iter_time_plot = pd.DataFrame()
        iter_time_plot['mean iterations num'] = policy_iteration_mean_iter + value_iteration_mean_iter
        iter_time_plot['mean time (s)'] = policy_iteration_mean_time + value_iteration_mean_time
        iter_time_plot['color'] = ['policy_iteration_' + a for a in action_set] + \
                                  ['value_iteration_' + a for a in action_set]
        iter_time_plot['map name'] = map_name + map_name

        fig = px.line(iter_time_plot, x='map name', y='mean iterations num', color='color', symbol="color")
        fig.update_traces(textposition="bottom right")
        fig.update_layout(title='Convergence time comparison')

        if self.cfg.save_plots:
            fig.write_image(self.cfg.plots_dir + f"Convergence time comparison.png")
        fig.show()

        fig = px.line(iter_time_plot, x='map name', y='mean time (s)', color='color', symbol="color")
        fig.update_traces(textposition="bottom right")
        fig.update_layout(title='Run time comparison')

        if self.cfg.save_plots:
            fig.write_image(self.cfg.plots_dir + f"Run time comparison.png")
        fig.show()
