import os
import datetime
from sqlalchemy import create_engine

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from Databases.DataScratching.uploadProceduresNBP import engine_create

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False