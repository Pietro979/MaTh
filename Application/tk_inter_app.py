from tkinter import *
from tkinter import ttk
from tk_inter_app_lib import PredictionsApp


root = Tk()
PredictionsApp(root)
root.mainloop()


# root.title("Feet to Meters")
# root.geometry("500x400")
# root.resizable(True,True)
#
# mainframe = ttk.Frame(root)
# mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
# root.columnconfigure(0, weight=1)
# root.rowconfigure(0, weight=1)
#
#
# n = ttk.Notebook(mainframe)
# f1 = ttk.Frame(n)   # first page, which would get widgets gridded into it
# f2 = ttk.Frame(n)   # second page
# n.add(f1, text='Single Step')
# n.add(f2, text='Multiple Steps')
#
# for child in mainframe.winfo_children():
#     child.grid_configure(padx=5,pady=5)
#
#
# root.mainloop()