import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_water


#df = pd.read_csv("datasets/water.csv")

#print(df.head())
#print(df.info())

water = load_water()


x = water.data[:]
y = water.target

print(x)
print("=====================================================")
print("=====================================================\n")
print(y)