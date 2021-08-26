import matplotlib.pyplot as plt

from CurrencyPrediction.Predictions.prepareData import PrepareData
import pandas as pd
import numpy as np
import tensorflow as tf
from CurrencyPrediction.Predictions.prepareData import *


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, df, train_mean=None, train_std = None,
                 label_columns=None, single_pred=False):
        # Store the raw data.
        # global test_df
        # global df
        self.train_mean = train_mean
        self.train_std = train_std
        test_df = test_df.loc[(test_df != 0).any(1)]
        df = df.loc[(df != 0).any(1)]

        if single_pred:
            zeros = pd.DataFrame(data=[0], columns=['Value'])
            test_df = test_df.append(other=zeros, ignore_index=True)
            df = df.append(other=zeros, ignore_index=True)

        else:
            zeros = pd.DataFrame(np.zeros((shift, 1)), columns=['Value'])
            test_df = test_df.append(other=zeros, ignore_index=True)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


def split_window(self, features, real_values = None):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    self.real_values = real_values

    return inputs, labels


WindowGenerator.split_window = split_window


def plot(self, model=None, plot_col='Value', max_subplots=3, last=True, future=True):
    result = getattr(self, '_example', None)
    if result is None:
        inputs, labels = self.example
    else:
        inputs, labels = self._example

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        # plt.plot(self.input_indices, inputs[n, :, plot_col_index],
        #          label='Inputs', marker='.', zorder=-10)
        plt.plot(self.real_values[:-1, 0], self.real_values[:-1, 1])

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue
        # małe oszustwo
        if last == False:
            # plt.scatter(self.label_indices[:-self.shift], labels[n, :-self.shift, label_col_index],  # bez ostatniego
            #             edgecolors='k', label='Labels', c='#2ca02c', s=64)
            plt.scatter(self.real_values[:-1, 0], self.real_values[:-1, 1],  # bez ostatniego
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
        else:
            # plt.scatter(self.label_indices[:], labels[n, :, label_col_index],  # z ostatnim
            #             edgecolors='k', label='Labels', c='#2ca02c', s=64)
            plt.scatter(self.real_values[:, 0], self.real_values[:, 1],  # bez ostatniego
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

        if model is not None:
            predictions = model(inputs)
            predictions = (predictions * self.train_std) + self.train_mean
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')
    plt.show()


WindowGenerator.plot = plot


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        #     # No example batch was found, so get one from the `.train` dataset
        #         result = next(iter(self.train))
        #     # And cache it for next time
        #         self._example = result
        example_window = tf.stack([np.array(self.test_df[-self.total_window_size:])]) #TUTAJ COŚ DODALEM(!self!.test_df)
        example_inputs, example_labels = self.split_window(example_window)
        self._example = example_inputs, example_labels
        result = self._example
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
