from tkinter import *
from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame


class PredictionsApp:

    def plot_data(self):
        data = [5, 7, 2, 5, 3, 6, 2, 1, 6, 4, 3, 6, 8]
        df = DataFrame(data, columns=['First Column Name'])

        figure1 = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax1 = figure1.gca()
        self.plot1 = FigureCanvasTkAgg(figure1, self.chart_frame)
        self.plot1.get_tk_widget().pack()
        df.plot(kind='line',
                ax=self.ax1,
                legend=False)

    def __init__(self, root):
        root.title("PredictionApp")
        root.geometry("600x400")
        root.resizable(True,True)
        self.options_width = 0.25
        self.options_height = 0.2
        self.chart_width = 0.75

        notebook = ttk.Notebook(root)
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
        self.dates_frame.place(relx=self.chart_width, rely=self.options_height*0, relheight=self.options_height, relwidth=self.options_width)
        self.rates_frame.place(relx=self.chart_width, rely=self.options_height*1, relheight=self.options_height, relwidth=self.options_width)
        self.method_frame.place(relx=self.chart_width, rely=self.options_height*2, relheight=self.options_height, relwidth=self.options_width)
        self.predict_button_frame.place(relx=self.chart_width, rely=self.options_height*3, relheight=self.options_height, relwidth=self.options_width)
        self.result_frame.place(relx=self.chart_width, rely=self.options_height*4, relheight=self.options_height, relwidth=self.options_width)

        self.predict_button = ttk.Button(self.predict_button_frame, text = "Predict", command = self.plot_data)
        self.predict_button.place(relx = 0.5, rely = 0.5, anchor = CENTER, relheight = 0.9, relwidth = 0.9)

        self.date_from_str = StringVar()
        self.date_from_str_cb = ttk.Combobox(self.dates_frame, textvariable=self.date_from_str)
        self.date_from_str_cb.place(relx=0.4, rely=0, relheight=0.3, relwidth=0.55)
        self.label_from = ttk.Label(self.dates_frame, text = "from: ")
        self.label_from.place(relx=0.05, rely=0, relheight=0.3, relwidth=0.35)
        self.label_to = ttk.Label(self.dates_frame, text="to: ")
        self.label_to.place(relx=0.05, rely=0.5, relheight=0.3, relwidth=0.35)

        self.date_to_str = StringVar()
        self.date_to_str_cb = ttk.Combobox(self.dates_frame, textvariable=self.date_to_str)
        self.date_to_str_cb.place(relx=0.4, rely=0.5, relheight=0.3, relwidth=0.55)

        self.rate_str = StringVar()
        self.rate_str_cb = ttk.Combobox(self.rates_frame, textvariable=self.rate_str)
        self.rate_str_cb.place(relx = 0.5, rely = 0.5, anchor = CENTER, relheight = 0.9, relwidth = 0.9)

        self.method_str = StringVar()
        self.method_str_cb = ttk.Combobox(self.method_frame, textvariable=self.method_str)
        self.method_str_cb.place(relx=0.5, rely=0.5, anchor=CENTER, relheight=0.9, relwidth=0.9)