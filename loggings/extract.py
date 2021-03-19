import os
import shutil
import re
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
import json

from configs import LogsOptions
from loggings.logger import Logger

class LogsExtractor():
    def __init__(self, logger: Logger, options: LogsOptions):
        self.logger = logger
        self.options = options
        self()

    def __call__(self):
        logs_dir = sorted([os.path.join(self.options.logs_dir, log_dir) for log_dir in os.listdir(self.options.logs_dir) if not log_dir.startswith('.')])
        names = sorted([log_dir for log_dir in os.listdir(self.options.logs_dir) if not log_dir.startswith('.')])
        
        # Create CSVs
        aggregations = self.get_experiments(logs_dir, names, prune=True)
        for aggregation in aggregations:
            self.aggregate_csvs(**aggregation)

        # Create plots
        aggregations = self.get_experiments(logs_dir, names, prune=False)
        for aggregation in aggregations:
            self.aggregate_plots(**aggregation)


    def get_experiments(self, logs_dir, names, prune=False):
        aggregations = []
        for log_dir, name in zip(logs_dir, names):
            events_dir = os.path.join(log_dir, 'logs')
            events_test_dir = os.path.join(events_dir, 'test')
            has_events_test_dir = os.path.isdir(events_test_dir)
            aggregations.append({
                'name': name,
                'source_dir': log_dir,
                'events': sorted([os.path.join(events_dir, event) for event in os.listdir(events_dir) if 'tfevents' in event]),
                'csv_dir': os.path.join(log_dir, 'csv'),
                'plot_dir': os.path.join(log_dir, 'plots')
            })
            if prune and has_events_test_dir:
                aggregations.append({
                    'name': name,
                    'source_dir': log_dir,
                    'events': sorted([os.path.join(events_test_dir, event) for event in os.listdir(events_test_dir) if 'tfevents' in event]),
                    'csv_dir': os.path.join(log_dir, 'csv_test'),
                    'plot_dir': os.path.join(log_dir, 'plots')
                })
        return aggregations


    def aggregate_csvs(self, name, events, source_dir, csv_dir, plot_dir):
        already_processed = os.path.isdir(csv_dir)
        if not already_processed:
            # Extract scalars from event files
            extracts = self.extract(events)
            # Create csv
            self.aggregate_to_csv(name, extracts, csv_dir)
            self.logger.log_info(f'Aggregation finished: {name}')


    def write_plot(self, csv_dir, plot_dir, keys, legends, colors, xlabel, ylabel, concat=False, smoothing=False, size=(500,500), grid=True, sci_ticks=True, ignore_outliers=True, smooth={'rolling':15, 'alpha':0.33}, filename=None, format='.png'):
        legend_handles, ylower, yupper = [], [], []
        dpi = 100
        figsize = (size[0] / dpi, size[1] / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)

        assert len(keys) == len(colors) == len(legends), 'Lists: keys, legends, and colors must have the same lengths!'

        # Format scientific notation
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(sci_ticks) 
        formatter.set_powerlimits((-1,1)) 

        if concat:
            fig, axes = plt.subplots(1, len(keys), figsize=figsize, dpi=dpi, sharey=True, sharex=False)

        for i, (key, legend, color) in enumerate(zip(keys, legends, colors)):
            # Combine multiple CSVs by operation, e.g.: csv_name1:csv_name2:mean
            if ':' in key and len(key.split(':')) > 0:
                agg_keys = key.split(':')[0:-1]
                agg_op = key.split(':')[-1]
                data_aggs = [self.get_data_frame(csv_dir, ag_key) for ag_key in agg_keys]
                data = pd.concat(data_aggs)
                by_row_index = data.groupby(data.index)
                data = getattr(by_row_index, agg_op)()
            # Process single CSV
            else:
                data = self.get_data_frame(csv_dir, key)

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

            # Set labels
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.yaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(grid)
                
            # Add legend
            legend_handles.append(mpatches.Patch(color=color, label=legend))
            plt.legend(handles=legend_handles, loc='best')

        # Ignore outliers
        if ignore_outliers:
            plt.ylim(min(ylower), max(yupper))

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save plot
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        filename = self.get_valid_filename(filename.lower()) + format
        fig.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
        plt.close(fig)


    def get_data_frame(self, csv_dir, key, format='.csv'):
        df = pd.read_csv(os.path.join(csv_dir, key+format), sep=r',', header=0, index_col='step')
        return df['value']


    def aggregate_plots(self, name, events, source_dir, csv_dir, plot_dir):
        with open('./loggings/plots.json') as f:
            plots = json.load(f)
            plots_config = plots['config']
            plots = plots['plots']
        
        for plot in plots:
            self.write_plot(csv_dir, plot_dir, **plot, **plots_config)

        self.logger.log_info(f'Plots created: {name}')


    def extract(self, events):
        accumulators = [EventAccumulator(event).Reload().scalars for event in events]
        # Filter non event files
        accumulators = [accumulator for accumulator in accumulators if accumulator.Keys()]
        # Get and validate all scalar keys
        all_keys = [tuple(accumulator.Keys()) for accumulator in accumulators]
        keys = all_keys[0]

        all_scalar_events_per_key = [[accumulator.Items(key) for accumulator in accumulators if key in accumulator.Keys()] for key in keys]
        all_scalars_accumulated = []

        for scalar_events_per_key in all_scalar_events_per_key:
            accumulated = []
            for scalar_events in scalar_events_per_key:
                accumulated = accumulated + scalar_events

            scalar_events_per_key = [[acc.step, acc.wall_time, acc.value] for acc in accumulated]
            all_scalars_accumulated.append(scalar_events_per_key)

        all_per_key = dict(zip(keys, all_scalars_accumulated))
        return all_per_key


    def aggregate_to_csv(self, name, extracts, csv_dir):
        for key, all_per_key in extracts.items():
            self.write_csv(csv_dir, key, name, all_per_key)
        # Move CSVs in 'csv_test' to 'csv' directory
        if 'csv_test' in csv_dir:
            csvs_test = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if not file.startswith('.')]
            for csv_test in csvs_test:
                shutil.move(csv_test, csv_test.replace('csv_test', 'csv'))
            # TODO: uncomment
            #os.rmdir(csv_dir)


    def write_csv(self, csv_dir, key, name, aggregations):
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        
        filename = f'{self.get_valid_filename(key.lower())}.csv'
        aggregations = np.asarray(aggregations)
        df = pd.DataFrame(aggregations, index=None, columns=['step', 'wall_time', 'value'])
        df.to_csv(os.path.join(csv_dir, filename), sep=',')


    def get_valid_filename(self, s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)
