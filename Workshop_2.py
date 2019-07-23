import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read the data into a dataframe
data = pd.read_csv("E:/python/WorkShop_1/goodreads.csv")

#Examine the first couple of rows of the dataframe
print(data.head())

data = pd.read_csv("E:/python/WorkShop_1/goodreads.csv", header=None,
               names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],
)
#Examine the first couple of rows of the dataframe
print(data.head())

print(data.dtypes)

print(data.shape)
print(data.columns)



#Get a sense of how many missing values there are in the dataframe.
print(np.sum([data.rating.isnull()]))
# do this for other columns as well

#Try to locate where the missing values occur
print(data[data.rating.isnull()])

print(data[data.year.isnull()])

#Treat the missing or invalid values in your dataframe
data = data[data.year.notnull()]

print(np.sum(data.year.isnull()))
print(np.sum(data.rating_count.isnull()))
print(np.sum(data.review_count.isnull()))
# How many rows are removed?
print(data.shape)

#lets try to change the data types of rating count and year to integer
data.rating_count=data.rating_count.astype(int)
data.review_count=data.review_count.astype(int)
data.year=data.year.astype(int)

print(data.dtypes)



# Some of the other colums that should be strings have NaN.
data.loc[data.genre_urls.isnull(), 'genre_urls']=""
data.loc[data.isbn.isnull(), 'isbn']=""

print("------- Asking and Answering questions ------------")

#Get the first author_url
test_string = data.author_url[0]
print(test_string)

# Isolate the author name
print(test_string.split('/')[-1].split('.')[1:][0])


# Write a function that accepts an author url and returns the author's name based on your experimentation above
def get_author(url):
    name = url.split('/')[-1].split('.')[1:][0]
    return name


# Apply the get_author function to the 'author_url' column using '.map'
# and add a new column 'author' to store the names
data['author'] = data.author_url.map(get_author)
print(data.author[0:5])

# Now parse out the genres from genre_url. Like with the authors, we'll assign a new genres column to the dataframe. This is a little more complicated because there be more than one genre.

print(data.genre_urls.head())

#Examine some examples of genre_urls
#Test out some string operations to isolate the genre name
test_genre_string = data.genre_urls[0]
print("dekhi : " + test_genre_string)

genres = test_genre_string.strip().split('|')
print("dekhi : ")
print(genres)

for e in genres:
    print(e.split('/')[-1])
# "|".join(genres)

#Write a function that accepts a genre url and returns the genre name based on your experimentation above
def split_and_join_genres(url):
    genres = url.strip().split('|')
    genres = [e.split('/')[-1] for e in genres]
    return "|".join(genres)

data['genres'] = data.genre_urls.map(split_and_join_genres)
data.head()

# test the function
split_and_join_genres("")
print(split_and_join_genres("/genres/young-adult|/genres/science-fiction"))

print(data[data.author == "Marguerite_Yourcenar"])


print("------ Part 4: EDA ----------")

# Generate histograms using the format data.COLUMN_NAME.hist(bins=YOUR_CHOICE_OF_BIN_SIZE)
# If your histograms appear strange or counter-intuitive, make appropriate adjustments in the data and re-visualize.

# data.review_count.hist(bins=200)
# plt.xlabel('Number of reviews')
# plt.ylabel('Frequency')
# plt.title('Number of reviews')
#
# plt.show()

# plt.hist(data.year, bins=100)
# plt.xlabel('Year written')
# plt.ylabel('log(Frequency)')
# plt.title('Number of books in a year')
# plt.show()

#It appears that some books were written in negative years!
# Print out the observations that correspond to negative years.
print(data[data.year < 0].name)
# What do you notice about these books?

print("-------- Part 5: Determining the Best Books -------")

#Using .groupby, we can divide the dataframe into subsets by the values of 'year'.
#We can then iterate over these subsets

for year, subset in data.groupby('year'):
    #Find the best book of the year
    bestbook = subset[subset.rating == subset.rating.max()]
    if bestbook.shape[0] > 1:
        print(year, bestbook.name.values, bestbook.rating.values)
    else:
        print(year, bestbook.name.values[0], bestbook.rating.values[0])

print("----- Part 6: Trends in Popularity of Genres ----")

#Get the unique genres contained in the dataframe.
genres = set()
for genre_string in data.genres:
    genres.update(genre_string.split('|'))
genres = sorted(genres)
print(genres)

# Add a column for each genre
for genre in genres:
    data["genre:" + genre] = [genre in g.split('|') for g in data.genres]

print(data.head())

print(data.shape)

print("----- encoding change  --------")

genreslist = ['genre:'+g for g in genres]
dfg = data[genreslist].sum() # True's sum as 1's, and default sum is columnwise

dfg.sort_values(ascending=False)

dfg.sort_values(ascending=False).plot(kind = "bar")

# The above histogram looks very clumsy!
# so now view less data
dfg.sort_values(ascending=False).iloc[0:20].plot(kind = "bar")

print(dfg.sort_values(ascending=False)[0:10])

genres_wanted=dfg.index[dfg.values > 550]
print(genres_wanted.shape)
print(genres_wanted)

print("-------- Shesh ------------")

fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(12, 40), tight_layout=True)
bins = np.arange(1950, 2013, 3)
for ax, genre in zip(axes.ravel(), genres_wanted):
    ax.hist(data[data[genre] == True].year.values, bins=bins, histtype='stepfilled', density=True, color='r', alpha=.2,
            ec='none')
    ax.hist(data.year, bins=bins, histtype='stepfilled', ec='None', density=True, zorder=0, color='#cccccc')

    ax.annotate(genre.split(':')[-1], xy=(1955, 3e-2), fontsize=14)
    ax.xaxis.set_ticks(np.arange(1950, 2013, 30))