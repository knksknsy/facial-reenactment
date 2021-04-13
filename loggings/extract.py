import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from configs import LogsOptions
from utils.utils import Method
from loggings.logger import Logger

class LogsExtractor():
    def __init__(self, logger: Logger, options: LogsOptions, experiments_path:str, multiples: bool, video_per_model: bool, method: Method):
        self.logger = logger
        self.options = options
        self.experiments_path = experiments_path
        self.video_per_model = video_per_model
        self.multiples = multiples
        self.method = method
        self.overwrite_csv = self.options.overwrite_csv
        self.overwrite_plot = self.options.overwrite_plot
        self.overwrite_video = self.options.overwrite_video

    def start(self):
        if not self.multiples:
            experiment_paths = [os.path.sep.join(self.experiments_path.split(os.path.sep)[:-1])]
            experiment_names = [self.experiments_path.split(os.path.sep)[-2]]
        else:
            experiment_paths = sorted([os.path.join(self.experiments_path, d) for d in os.listdir(self.experiments_path) if not d.startswith('.')])
            experiment_names = sorted([d for d in os.listdir(self.experiments_path) if not d.startswith('.')])
        
        aggregations = self.get_meta_data(experiment_paths, experiment_names)
        for aggregation in aggregations:
            # Create CSVs
            if self.overwrite_csv or not os.path.isdir(aggregation['csv_path']):
                self.aggregate_csv(**aggregation)

            # Create plots
            if self.overwrite_plot or not os.path.isdir(aggregation['plot_path']):
                self.aggregate_plots(**aggregation)

            # Create video
            if self.overwrite_video:
                self.save_video(**aggregation)


    def get_meta_data(self, experiment_paths, names):
        aggregations = []
        for ep, name in zip(experiment_paths, names):
            logs_dir = os.path.join(ep, 'logs')
            aggregations.append({
                'name': name,
                'experiment_path': ep,
                'event_paths': sorted([os.path.join(logs_dir, log) for log in os.listdir(logs_dir) if 'tfevents' in log]),
                'csv_path': os.path.join(ep, 'csv'),
                'plot_path': os.path.join(ep, 'plots'),
                'checkpoints_path': os.path.join(ep, 'checkpoints'),
                'video_path': os.path.join(ep, 'outputs')
            })
        return aggregations


    def aggregate_csv(self, name, event_paths, experiment_path, csv_path, plot_path, checkpoints_path, video_path):
        self.logger.log_info(f'Creating CSVs for experiment "{name}"...')
        # Extract scalars from event files
        all_extracts = self.extract(event_paths)
        for extracts in all_extracts:
            for key, values in extracts.items():
                filename = f'{self.get_valid_filename(key.lower())}.csv'
                self.write_csv(csv_path, filename, values)
                self.logger.log_info(f'CSV {filename} created.')

        self.logger.log_info(f'CSVs created for experiment: "{name}" into: {experiment_path}')


    def extract(self, event_paths):
        accumulators = [EventAccumulator(event).Reload().scalars for event in event_paths]
        # Filter non event files
        accumulators = [accumulator for accumulator in accumulators if accumulator.Keys()]

        # Get and validate all scalar keys
        all_keys = [tuple(accumulator.Keys()) for accumulator in accumulators]
        unique_all_keys = set()
        for key in all_keys:
            unique_all_keys.add(key)
        
        all_per_key_list = []
        for keys in unique_all_keys:
            all_scalar_events_per_key = [[accumulator.Items(key) for accumulator in accumulators if key in accumulator.Keys()] for key in keys]
            all_scalars_accumulated = []

            for scalar_events_per_key in all_scalar_events_per_key:
                accumulated = []
                for scalar_events in scalar_events_per_key:
                    accumulated = accumulated + scalar_events

                scalar_events_per_key = [[acc.step, acc.wall_time, acc.value] for acc in accumulated]
                all_scalars_accumulated.append(scalar_events_per_key)

            all_per_key = dict(zip(keys, all_scalars_accumulated))
            all_per_key_list.append(all_per_key)
        
        return all_per_key_list


    def write_csv(self, csv_path, filename, aggregations):
        if not os.path.isdir(csv_path):
            os.makedirs(csv_path)
        
        aggregations = np.asarray(aggregations)
        df = pd.DataFrame(aggregations, index=None, columns=['step', 'wall_time', 'value'])
        df.to_csv(os.path.join(csv_path, filename), sep=',')


    def aggregate_plots(self, name, event_paths, experiment_path, csv_path, plot_path, checkpoints_path, video_path):
        self.logger.log_info(f'Creating plots for experiment "{name}"...')
        plots_config = self.options.plots['config']
        plots = self.options.plots['plots']
        
        for plot in plots:
            self.write_plot(csv_path, plot_path, **plot, **plots_config)
            self.logger.log_info(f'Plot {plot["filename"]} created.')

        self.logger.log_info(f'Plots created for experiment "{name}" into: {experiment_path}')


    def write_plot(self, csv_path, plot_path, keys, legends, colors, xlabel, ylabel, concat=False, smoothing=False, size=(500,500), grid=True, sci_ticks=True, sci_format=None, ignore_outliers=True, smooth={'rolling':15, 'alpha':0.33}, filename=None, format='.png'):
        legend_handles, ylower, yupper = [], [], []
        dpi = 100
        figsize = (size[0] / dpi, size[1] / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)

        assert len(keys) == len(colors) == len(legends), 'Lists: keys, legends, and colors must have the same lengths!'

        if concat:
            fig, axes = plt.subplots(1, len(keys), figsize=figsize, dpi=dpi, sharey=True, sharex=False)

        for i, (key, legend, color) in enumerate(zip(keys, legends, colors)):
            # Combine multiple CSVs by operation, e.g.: csv_name1:csv_name2:mean
            if ':' in key and len(key.split(':')) > 0:
                agg_keys = key.split(':')[0:-1]
                agg_op = key.split(':')[-1]
                data_aggs = []
                for ag_key in agg_keys:
                    d = self.get_data_frame(csv_path, ag_key)
                    if d is None:
                        return
                    data_aggs.append(d)
                data = pd.concat(data_aggs)
                by_row_index = data.groupby(data.index)
                data = getattr(by_row_index, agg_op)()
            # Process single CSV
            else:
                data = self.get_data_frame(csv_path, key)
                if data is None:
                    return

            # Get outliers
            if ignore_outliers:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                ylower.append(q1 - (1.5 * iqr))
                yupper.append(q3 + (1.5 * iqr))

            if not concat:
                ax = plt.subplot(111)
            else:
                ax = axes[i]
            
            # Plot chart
            if smoothing:
                data.plot(color=color, alpha=smooth['alpha'], ax=ax)
                data.rolling(smooth['rolling']).mean().plot(color=color, ax=ax)
            else:
                data.plot(color=color, ax=ax)

            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.grid(grid)
            
            # Format scientific notation
            formatter = ticker.ScalarFormatter(useMathText=True)
            if sci_format is None:
                formatter.set_scientific(sci_ticks)
            else:
                formatter.set_scientific(sci_format)
            formatter.set_powerlimits((-1,1))
            ax.yaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(formatter)

            # Add legend
            legend_handles.append(mpatches.Patch(color=color, label=legend))
            plt.legend(handles=legend_handles, loc='best')

        # Ignore outliers
        if ignore_outliers:
            plt.ylim(min(ylower), max(yupper))

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save plot
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)
        filename = self.get_valid_filename(filename.lower()) + format
        fig.savefig(os.path.join(plot_path, filename), bbox_inches='tight')
        plt.close(fig)


    def get_data_frame(self, csv_path, key, format='.csv'):
        if not os.path.isfile(os.path.join(csv_path, key+format)):
            return None
        df = pd.read_csv(os.path.join(csv_path, key+format), sep=r',', header=0, index_col=None)
        df = df.sort_values(by='step', ascending=True)
        df = df.drop_duplicates(subset='step', keep='last')
        df = df.set_index('step')
        return df['value']


    def get_valid_filename(self, s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)


    def save_video(self, name, event_paths, experiment_path, csv_path, plot_path, checkpoints_path, video_path):
        if self.method == Method.CREATION:
            from creation.inference import Infer
        elif self.method == Method.DETECTION:
            from detection.inference import Infer

        if not os.path.isdir(video_path):
            os.makedirs(video_path)

        # Create video for each checkpoint
        model_paths = [f for f in os.listdir(checkpoints_path) if not f.startswith('.')]
        if self.video_per_model:
            model_paths = sorted([os.path.join(checkpoints_path, f) for f in model_paths if not f.startswith('.') and 'Generator_' in f])
            for i, m in enumerate(model_paths):
                infer = Infer(self.logger, self.options, self.options.v_img_source, self.options.v_vid_target, m)
                infer.from_video(filename=f'e{str(i).zfill(3)}')
        
        # Create video for latest and best checkpoint
        # Latest checkpoint
        model_paths = [f for f in os.listdir(checkpoints_path) if not f.startswith('.')]
        latest_epoch = (len(model_paths)//2) - 1
        model_path = [os.path.join(checkpoints_path, f) for f in model_paths if f'_e{str(latest_epoch).zfill(3)}' in f and 'Generator_' in f][0]
        infer = Infer(self.logger, self.options, self.options.v_img_source, self.options.v_vid_target, model_path)
        infer.from_video(filename=f'latest_e{str(latest_epoch).zfill(3)}', output_path=video_path)

        # Best checkpoint
        df = pd.read_csv(os.path.join(csv_path, 'fid_validation.csv'), sep=r',', header=0, index_col=None)
        df = df.sort_values(by='step', ascending=True)
        df = df.drop_duplicates(subset='step', keep='last')
        min_epoch = int(df.loc[df['value'].idxmin()]['step'])
        model_path = [os.path.join(checkpoints_path, f) for f in model_paths if f'_e{str(min_epoch).zfill(3)}' in f and 'Generator_' in f][0]
        infer = Infer(self.logger, self.options, self.options.v_img_source, self.options.v_vid_target, model_path)
        infer.from_video(filename=f'best_e{str(min_epoch).zfill(3)}', output_path=video_path)
