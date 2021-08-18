import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame

data = [5,7,2,5,3,6,2,1,6,4,3,6,8]
figure = plt.Figure(figsize=(6,5), dpi=100)
ax = figure.add_subplot(111)
df = DataFrame(data, columns=['First Column Name'])
df.plot()
plt.show()