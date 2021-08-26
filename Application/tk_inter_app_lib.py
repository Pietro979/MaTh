from tkinter import *
from tkinter import ttk
from CurrencyPrediction.Predictions.predictionModels import *
from CurrencyPrediction.Predictions.WindowGenerator import *
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame
from matplotlib.ticker import FixedLocator, FixedFormatter
import os
import datetime
from CurrencyPrediction.Predictions.prepareData import PrepareData
from datetime import timedelta
from sqlalchemy import create_engine

import IPython
import IPython.display
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sqlalchemy import select, MetaData, Table, insert, Column, String, Sequence, Integer, Float
import matplotlib.pyplot as plt

from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey, and_
from Databases.DataScratching.uploadProceduresNBP import engine_create
import Databases.ConnectDatabase
from Databases.ConnectDatabase import ConnectDatabase
from CurrencyPrediction.Predictions.predictionModels import *


class PredictionsApp:
    def choose_data(self):
        year = self.date_from_year.get()
        month = self.date_from_month.get()
        day = self.date_from_day.get()
        days = self.how_many_days_entry.get()
        start_date = datetime.date(int(year), int(month), int(day))
        rate = self.rates_dict[self.rate_str_cb.get()]

        global OUT_STEPS


        if 'Multistep' in self.method_str_cb.get():
            self.chosen_data = self.database.choose_data(start_date, 2*OUT_STEPS, rate)
            print('Multistep')
        else:
            self.chosen_data = self.database.choose_data(start_date, int(days), rate)
            print('not Multistep')
        # df = DataFrame(data, columns=['First Column Name'])
        # print(df)
        # if self.ax1 is not None:
        #     self.ax1.clear()
        #
        # x_formatter = FixedFormatter([, self.date_to.strftime('%d %b %Y')])
        # x_locator = FixedLocator([0, len(df['Value'])])
        #
        # figure1 = plt.Figure(figsize=(6, 5), dpi=100)
        # self.ax1 = figure1.gca()
        #
        # self.plot1 = FigureCanvasTkAgg(figure1, self.chart_frame)
        # self.plot1.get_tk_widget().pack()
        #
        # df.plot(kind='line',
        #         ax=self.ax1,
        #         legend=False)

    def train_data(self):
        prepare_data = PrepareData(self.database, rate_id = self.rates_dict[self.rate_str_cb.get()])

        self.train_df = prepare_data.train_df
        self.val_df = prepare_data.val_df
        self.test_df = prepare_data.test_df
        self.df_std = prepare_data.df_std
        self.column_indices = prepare_data.column_indices
        self.train_mean = prepare_data.train_mean
        self.train_std = prepare_data.train_std

        if self.method_str_cb.get() == 'Conv Neural Network':
            conv_window = WindowGenerator(
                input_width=CONV_WIDTH,
                label_width=1,
                shift=1, train_df=self.train_df, test_df=self.test_df,
                val_df=self.val_df, df=self.df_std,
                label_columns=['Value'])
        elif 'Multistep' in self.method_str_cb.get():
            self.window = WindowGenerator(input_width=OUT_STEPS,
                                           label_width=OUT_STEPS,
                                           shift=OUT_STEPS, train_df=self.train_df, test_df=self.test_df,
                                           val_df=self.val_df, df=self.df_std, train_mean=self.train_mean,
                                           train_std=self.train_std,
                                           single_pred=False)
        elif self.method_str_cb.get() == 'Recurrent Neural Network':
            wide_window = WindowGenerator(
                input_width=24,
                label_width=24,
                shift=1, train_df=self.train_df, test_df=self.test_df,
                val_df=self.val_df, df=self.df_std, train_mean=self.train_mean, train_std=self.train_std,
                label_columns=['Value'], single_pred=True)
        else:
            single_step_window = WindowGenerator(
                input_width=1, label_width=1, shift=1, single_pred=True, train_df=self.train_df, test_df=self.test_df,
                val_df=self.val_df, df=self.df_std,
                label_columns=['Value'])

        if self.method_str_cb.get() == 'Baseline':
            self.baseline = Baseline(label_index=self.column_indices['Value'])
            self.baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
        elif self.method_str_cb.get() == 'Linear Model':
            self.history = compile_and_fit(linear, single_step_window)
        elif self.method_str_cb.get() == 'Dense':
            self.history = compile_and_fit(dense, single_step_window)
        elif self.method_str_cb.get() == 'Many Step Dense':
            self.history = compile_and_fit(multi_step_dense, conv_window)
        elif self.method_str_cb.get() == 'Conv Neural Network':
            self.history = compile_and_fit(conv_model, conv_window)
        elif self.method_str_cb.get() == 'Recurrent Neural Network':
            self.history = compile_and_fit(lstm_model, wide_window)
        elif self.method_str_cb.get() == 'Multistep baseline':
            self.last_baseline = MultiStepLastBaseline()
            self.last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                  metrics=[tf.metrics.MeanAbsoluteError()])
        elif self.method_str_cb.get() == 'Multistep Repeat baseline':
            self.repeat_baseline = RepeatBaseline()
            self.repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                    metrics=[tf.metrics.MeanAbsoluteError()])
        elif self.method_str_cb.get() == 'Multistep Linear Model':
            self.history = compile_and_fit(multi_linear_model, self.window)
        elif self.method_str_cb.get() == 'Multistep Dense':
            self.history = compile_and_fit(multi_dense_model, self.window)
        elif self.method_str_cb.get() == 'Multistep Conv Neural Network':
            self.history = compile_and_fit(multi_conv_model, self.window)
        elif self.method_str_cb.get() == 'Multistep Recurrent Neural Network':
            self.history = compile_and_fit(multi_lstm_model, self.window)

        #
        # wide_window = WindowGenerator(
        #     input_width=24, label_width=24, shift=1, train_df=train_df, test_df_in=test_df,
        #     val_df=val_df,
        #     label_columns=['Value'], single_pred=True)
        # wide_window.plot(linear, last=False)


        # val_performance = {}
        # performance = {}
        # val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
        # performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)


    def plot_data(self):

        pass

    def predict_data(self):
        data_to_present_np = np.array(self.chosen_data)
        data_to_predict_np = np.array((self.chosen_data['Value'] - self.train_mean['Value']) / self.train_std['Value'])
        data_to_predict_np = data_to_predict_np[:, np.newaxis]


        # wide_window
        if self.method_str_cb.get() in ['Baseline', 'Linear Model','Dense', 'Many Step Dense','Recurrent Neural Network']:
            wide_window = WindowGenerator(
                input_width=int(self.how_many_days_entry.get()) - 1, label_width=int(self.how_many_days_entry.get()) - 1,
                shift=1, train_df=self.train_df, test_df=self.test_df,
                val_df=self.val_df, df=self.df_std, train_mean=self.train_mean, train_std = self.train_std,
                label_columns=['Value'], single_pred=True)

            example_inputs, example_labels = wide_window.split_window(tf.stack([data_to_predict_np]),
                                                                      real_values = data_to_present_np)
            wide_window._example = example_inputs, example_labels

        if self.method_str_cb.get() == 'Conv Neural Network':
            wide_conv_window = WindowGenerator(
                input_width=int(self.how_many_days_entry.get()) -1,
                label_width=int(self.how_many_days_entry.get()) - CONV_WIDTH,
                shift=1, train_df=self.train_df, test_df=self.test_df,
                val_df=self.val_df, df=self.df_std, train_mean=self.train_mean, train_std = self.train_std,
                label_columns=['Value'], single_pred=True)

            example_inputs, example_labels = wide_conv_window.split_window(tf.stack([data_to_predict_np]),
                                                                      real_values=data_to_present_np)
            wide_conv_window._example = example_inputs, example_labels

        if 'Multistep' in self.method_str_cb.get():
            self.window = WindowGenerator(input_width=OUT_STEPS,
                                           label_width=OUT_STEPS,
                                           shift=OUT_STEPS, train_df=self.train_df, test_df=self.test_df,
                                            val_df=self.val_df, df=self.df_std, train_mean=self.train_mean,
                                           train_std = self.train_std,
                                           single_pred=False)
            example_inputs, example_labels = self.window.split_window(tf.stack([data_to_predict_np]),
                                                                        real_values=data_to_present_np)
            self.window._example = example_inputs, example_labels

        if self.method_str_cb.get() == 'Baseline':
            wide_window.plot(self.baseline, last=False)
        elif self.method_str_cb.get() == 'Linear Model':
            wide_window.plot(linear, last=False)
        elif self.method_str_cb.get() == 'Dense':
            wide_window.plot(dense, last=False)
        elif self.method_str_cb.get() == 'Many Step Dense':
            self.conv_window.plot(multi_step_dense, last=False)
        elif self.method_str_cb.get() == 'Conv Neural Network':
            wide_conv_window.plot(conv_model, last=False)
        elif self.method_str_cb.get() == 'Recurrent Neural Network':
            wide_window.plot(lstm_model, last=False)
        elif self.method_str_cb.get() == 'Multistep baseline':
            self.window.plot(self.last_baseline, last=False)
        elif self.method_str_cb.get() == 'Multistep Repeat baseline':
            self.window.plot(self.repeat_baseline, last=False)
        elif self.method_str_cb.get() == 'Multistep Linear Model':
            self.window.plot(multi_linear_model, last=False)
        elif self.method_str_cb.get() == 'Multistep Dense':
            self.window.plot(multi_dense_model, last=False)
        elif self.method_str_cb.get() == 'Multistep Conv Neural Network':
            self.window.plot(multi_conv_model, last=False)
        elif self.method_str_cb.get() == 'Multistep Recurrent Neural Network':
            self.window.plot(multi_lstm_model, last=False)

    def __init__(self):
        year_tab = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
        month_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        day_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,28, 29, 30, 31]
        methods_tab = ['Baseline', 'Linear Model', 'Dense','Many Step Dense', 'Conv Neural Network',
                       'Recurrent Neural Network','Multistep baseline','Multistep Repeat baseline', 'Multistep Linear Model', 'Multistep Dense', 'Multistep Conv Neural Network', 'Multistep Recurrent Neural Network']

        self.database = ConnectDatabase()


        # rates:
        self.rates_dict = {mapper_result[1]: mapper_result[0] for mapper_result in self.database.engine.execute(
            select([self.database.dic_table['rates'].columns.rate_id, self.database.dic_table['rates'].columns.rate]))
            .fetchall()}
        self.rates_tab = list(self.rates_dict.keys())

        self.root = Tk()
        self.root.title("PredictionApp")
        self.root.geometry("800x500")
        self.root.resizable(True,True)
        self.options_width = 0.25
        self.options_height = 0.2
        self.chart_width = 0.75

        notebook = ttk.Notebook(self.root)
        notebook.place(relx = 0, rely = 0, relheight = 1, relwidth = 1)
        f1 = ttk.Frame(notebook)   # first page, which would get widgets gridded into it
        f2 = ttk.Frame(notebook)   # second page
        notebook.add(f1, text='Single Step')
        notebook.add(f2, text='Multiple Steps')

        self.chart_frame = ttk.LabelFrame(f1, text = "chart_frame")
        self.dates_frame = ttk.LabelFrame(f1, text = "dates_frame")
        self.rates_frame = ttk.LabelFrame(f1, text="rates_frame")
        self.method_frame = ttk.LabelFrame(f1, text="method_frame")
        self.predict_button_frame = ttk.LabelFrame(f1, text="predict_button_frame")
        self.result_frame = ttk.LabelFrame(f1, text="result_frame")

        self.chart_frame.place(relx = 0, rely = 0, relheight = 1, relwidth = self.chart_width)
        self.method_frame.place(relx=self.chart_width, rely=self.options_height*0, relheight=self.options_height, relwidth=self.options_width)
        self.rates_frame.place(relx=self.chart_width, rely=self.options_height*1, relheight=self.options_height, relwidth=self.options_width)
        self.dates_frame.place(relx=self.chart_width, rely=self.options_height*2, relheight=self.options_height, relwidth=self.options_width)
        self.predict_button_frame.place(relx=self.chart_width, rely=self.options_height*3, relheight=self.options_height, relwidth=self.options_width)
        self.result_frame.place(relx=self.chart_width, rely=self.options_height*4, relheight=self.options_height, relwidth=self.options_width)

        self.choose_button = ttk.Button(self.dates_frame, text="Choose Data", command=self.choose_data)
        self.choose_button.place(relx=0.5, rely=0.81, anchor=CENTER, relheight=0.38, relwidth=0.9)

        self.predict_button = ttk.Button(self.predict_button_frame, text = "Predict", command = self.predict_data)
        self.predict_button.place(relx = 0.5, rely = 0.5, anchor = CENTER, relheight = 0.9, relwidth = 0.9)

        # DATES FROM

        self.date_from_year = ttk.Combobox(self.dates_frame, values = year_tab)
        self.date_from_year.place(relx=0.2, rely=0, relheight=0.3, relwidth=0.3)

        self.date_from_month = ttk.Combobox(self.dates_frame, values=month_tab)
        self.date_from_month.place(relx=0.51, rely=0, relheight=0.3, relwidth=0.24)

        self.date_from_day = ttk.Combobox(self.dates_frame, values=day_tab)
        self.date_from_day.place(relx=0.76, rely=0, relheight=0.3, relwidth=0.24)

        self.label_from = ttk.Label(self.dates_frame, text = "from: ")
        self.label_from.place(relx=0.01, rely=0, relheight=0.3, relwidth=0.18)

        self.label_to = ttk.Label(self.dates_frame, text="for: ")
        self.label_to.place(relx=0.01, rely=0.31, relheight=0.3, relwidth=0.18)

        self.label_to_2 = ttk.Label(self.dates_frame, text="days")
        self.label_to_2.place(relx=0.8, rely=0.31, relheight=0.3, relwidth=0.18)

        how_many_days_str = StringVar()
        self.how_many_days_entry = ttk.Entry(self.dates_frame, textvariable = how_many_days_str)
        self.how_many_days_entry.place(relx=0.4, rely=0.31, relheight=0.3, relwidth=0.4)

        self.ax1 = None
        self.rate = None
        self.chosen_data = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.df_std = None
        self.column_indices = None
        self.baseline = None
        self.history = None
        self.train_mean = None
        self.train_std = None

        self.val_performance = None
        self.performance = None
        self.window = None
        self.last_baseline = None
        self.repeat_baseline = None


        # DATES TO

        # self.date_to_year = ttk.Combobox(self.dates_frame, values=year_tab)
        # self.date_to_year.place(relx=0.2, rely=0.5, relheight=0.3, relwidth=0.3)
        #
        # self.date_to_month = ttk.Combobox(self.dates_frame, values=month_tab)
        # self.date_to_month.place(relx=0.51, rely=0.5, relheight=0.3, relwidth=0.24)
        #
        # self.date_to_day = ttk.Combobox(self.dates_frame, values=day_tab)
        # self.date_to_day.place(relx=0.76, rely=0.5, relheight=0.3, relwidth=0.24)
        #
        # self.label_to = ttk.Label(self.dates_frame, text="to: ")
        # self.label_to.place(relx=0.01, rely=0.5, relheight=0.3, relwidth=0.18)

        self.rate_str = StringVar()
        self.rate_str_cb = ttk.Combobox(self.rates_frame, textvariable=self.rate_str, values = self.rates_tab)
        self.rate_str_cb.place(relx = 0.5, rely = 0.25, anchor = CENTER, relheight = 0.4, relwidth = 0.9)

        self.train_button = ttk.Button(self.rates_frame, text = "Train", command = self.train_data)
        self.train_button.place(relx = 0.5, rely = 0.75, anchor = CENTER, relheight = 0.4, relwidth = 0.9)

        self.method_str = StringVar()
        self.method_str_cb = ttk.Combobox(self.method_frame, textvariable=self.method_str, values = methods_tab)
        self.method_str_cb.place(relx=0.5, rely=0.25, anchor=CENTER, relheight=0.4, relwidth=0.9)


        ###########################################################################################################################################3


        self.root.mainloop()


