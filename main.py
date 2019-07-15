import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import networkx as nx
# os.environ["PROJ_LIB"] = r"/home/hristocr/anaconda3/envs/jupyter-notebook/share/proj"
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston_data = load_boston()

# print(boston_data.DESCR)
x_data = pd.DataFrame(boston_data.data)
x_data.columns = boston_data.feature_names

model = LinearRegression()
y_data = boston_data.target

model = model.fit(x_data.values, y_data)


columns=boston_data.feature_names
coefs = model.coef_

columns.shape
coefs.shape

y_pred = model.predict(x_data)

model.score(x_data, y_data)

