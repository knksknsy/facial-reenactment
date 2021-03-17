import os
import re
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

        aggregations = []
        for log_dir, name in zip(logs_dir, names):
            events_dir = os.path.join(log_dir, 'logs')
            events_test_dir = os.path.join(events_dir, 'test')
            has_events_test_dir = os.path.isdir(events_test_dir)
            aggregations.append({
                'name': name,
                'source_dir': log_dir,
                'events': sorted([os.path.join(events_dir, event) for event in os.listdir(events_dir) if 'tfevents' in event]),
                'output_dir': os.path.join(log_dir, 'csv')
            })
            if has_events_test_dir:
                aggregations.append({
                    'name': name,
                    'source_dir': log_dir,
                    'events': sorted([os.path.join(events_test_dir, event) for event in os.listdir(events_test_dir) if 'tfevents' in event]),
                    'output_dir': os.path.join(log_dir, 'csv_test')
                })

        for aggregation in aggregations:
            self.aggregate(**aggregation)


    def aggregate(self, name, events, source_dir, output_dir):
        already_processed = os.path.isdir(output_dir)
        if not already_processed:
            # Extract scalars from event files
            extracts = self.extract(events)
            # Create csv
            self.aggregate_to_csv(name, extracts, output_dir)
            self.logger.log_info(f'Aggregation finished: {name}')
            # Plots (losses: train, val; metrics: fid, ssim)


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


    def aggregate_to_csv(self, name, extracts, output_dir):
        for key, all_per_key in extracts.items():
            self.write_csv(output_dir, key, name, all_per_key)


    def write_csv(self, output_dir, key, name, aggregations):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        filename = f'{self.get_valid_filename(name.lower())}_{self.get_valid_filename(key.lower())}.csv'
        aggregations = np.asarray(aggregations)
        df = pd.DataFrame(aggregations[:,1:], index=aggregations[:,0], columns=['wall_time', 'value'])
        df.to_csv(os.path.join(output_dir, filename), sep=',')


    def get_valid_filename(self, s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)
