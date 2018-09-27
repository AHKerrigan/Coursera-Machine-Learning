import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    return pd.read_csv("data/ex1data1.txt", sep = ",", header=None)

def scatterPlot(data):
    dataframe = pd.DataFrame(data)
    print(dataframe)
    dfp = dataframe.plot(x=0, y=1, kind='scatter')
    dfp.set_xlabel("Population")
    dfp.set_ylabel("Profit")
    plt.show()
