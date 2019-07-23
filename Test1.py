import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm # allows us easy access to colormaps
import matplotlib.pyplot as plt # sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
# sets up pandas table display
#pd.set_option('display.width', 500)
#pd.set_option('display.max_columns', 100)
#pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options

print("---- Introduction to pandas Start --------")

my_array = np.array([1, 2, 3, 4]) #create
print(len(my_array)) # length
print(my_array[0:4]) # slice
for ele in my_array: # loop
    print(ele)
print(my_array.mean()) # calcuate mean by method call
print(np.mean(my_array)) # calculate mean using numpy
print(np.ones(10)) # generates 10 floating point ones
np.ones(10, dtype='int') # generates 10 integer ones
print(np.zeros(10))
np.random.random(10) # uniform on [0,1]
# generate random numbers from a normal distribution with mean 0 and variance 1
normal_array = np.random.randn(1000)
print(normal_array)
print("The sample mean = %f standard devation = %f" %(np.mean(normal_array), np.std(normal_array)))
#numpy supports vector operations
first = np.ones(5)
second = np.ones(5)
first + second
print(first + second)
print(first + 1)
print(first*5)
print(5+7*first)
test = np.array([6, 17, 3, 4])
print(np.std(test))
# if you wanted the distribution N(5,7) you could do:
normal_5_7 = 5 + 7*normal_array
np.mean(normal_5_7), np.std(normal_5_7)

ones_2d = np.ones([3, 4]) # 3 x 4 array of ones
ones_2d.shape # show size
print(ones_2d.T) # trasnpose
print(np.sum(ones_2d)) # sum all elements

print("----------- Introduction to numpy End ---------------")

print("----------- Introduction to Pandas Start ---------------")

arrcars = np.genfromtxt('E:/python/WorkShop_1/venv/mtcars.csv', delimiter=',', skip_header=1, usecols=(1,2,3,4,5,6,7,8,9,10,11))
print(arrcars.shape)
print(arrcars[0:2]) # not very nice

# Read in the csv files using Panda
data = pd.read_csv("E:/python/WorkShop_1/venv/mtcars.csv")
print(type(data))
data = data.rename(columns={"Unnamed: 0": "name"})
print(data.head())
print(data.shape)
print(data.columns)
print(data.mpg)
print(data['name'])

#And we can produce a histogram from these values
plt.hist(data.mpg.values, bins=20);
plt.xlabel("mpg");
plt.ylabel("Frequency")
plt.title("Miles per Gallon");
plt.savefig("Histogram")
# plt.show()

# you can get a histogram using panda
data.mpg.hist(bins=20);
plt.xlabel("mpg");
plt.ylabel("Frequency")
plt.title("Miles per Gallon");
# plt.show()
print(data[['am', 'mpg']])
#Listiness property 1: set length
print("OOOOOOOOOOO")
print(data.shape)     # 12 columns, each of length 32
print(len(data))      # the number of rows in the dataframe, also the length of a series
print(len(data.mpg))  # the length of a series
# Listiness property 2: iteration via loops
# One consequence of the column-wise construction of dataframes is that you cannot easily iterate over the rows.
# Instead, we iterate over the columns.
for column in data:  # iterating iterates over column names though, like a dictionary
    print(column)

# Or we can call the attribute `columns`.
print(data.columns)

# We can iterate series in the same way that we iterate lists.
# Here we print out the number of cylinders for each of the 32 vehicles.
for element in data.cyl:
    print(element)
# you can iterate over rows by using `itertuples`.

#Listiness property 3: slice
print(list(data.index)) # index for the dataframe
print(data.cyl.index) # index for the cyl series

print("loacation & ilocation")

print(data.iloc[0:3])
print(data.loc[0:5])

print(data.iloc[2:5, 1:4])
print(data.loc[7:9,['mpg', 'cyl', 'disp'] ])

#add another column named 'maker' by parsing the first column
data['maker'] = data.name.apply(lambda x: x.split()[0])
data['maker']
#data.head()

column_1 = pd.Series(range(4))
column_2 = pd.Series(range(4,8))
table = pd.DataFrame({'col_1': column_1, 'col_2': column_2})
table = table.rename(columns={"col_1": "Col_1", "col_2":"Col_2"})
print(table)
# try this
#table = table.rename({0: "zero", 1: "one", 2: "two", 3: "three"})
#table


data.dtypes

# Categorical
data.maker.unique()
data.maker.describe()
av_mpg = data.groupby('maker').mpg.mean()

#query
data.mpg < 100
data[data.mpg < 100].head() #try other queries
data.query("10 <= mpg <= 50").head()
data.sort_values(by="mpg").head(3)
data[data.gear == 4]
data.mpg.max()
data.groupby("maker").describe()

print("Histograms")

data.mpg.plot.hist()
plt.xlabel("mpg")

plt.hist(data.mpg, bins=20)
plt.xlabel("mpg")
plt.ylabel("Frequency")
plt.title("Miles per Gallon")
# plt.show()

data.drat.plot.hist();
plt.xlabel("drat");
plt.ylabel("Frequency");
plt.title("Rear axle ratio (drat)");
# plt.show()
print("mean = ", data.drat.mean())

print("Scatter plots")

plt.scatter(data.wt, data.mpg); # you could also use plot and plot data as dots, try that.
plt.xlabel("weight");
plt.ylabel("miles per gallon");

sub_data = data[['wt', 'mpg']]
data_temp = sub_data.sort_values('wt')
plt.plot(data_temp.wt, data_temp.mpg, 'o-');

av_mpg.plot(kind="barh")
data.boxplot(column = 'mpg', by = 'am')
#pie chart
science = {
    'interest': ['Excited', 'Kind of interested', 'OK', 'Not great', 'Bored'],
    'before': [19, 25, 40, 5, 11],
    'after': [38, 30, 14, 6, 12]
}
dfscience = pd.DataFrame.from_dict(science).set_index("interest")[['before', 'after']]
fig, axs = plt.subplots(1,2, figsize = (8.5,4))
dfscience.before.plot(kind="pie", ax=axs[0], labels=None);
axs[0].legend(loc="upper left", ncol=5, labels=dfscience.index)
dfscience.after.plot(kind="pie", ax=axs[1], labels=None);

print("----------- Introduction to Pands End ---------------")