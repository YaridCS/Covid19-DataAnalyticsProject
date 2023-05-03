import io
import pandas as pd
import requests
import numpy as np

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
response = requests.get(url)
data = io.StringIO(response.content.decode('utf-8'))

df = pd.read_csv(data)


df.to_csv('covid19_datatable_original.csv', index=False)


# Isolating data from the columns of interest
df = df[['location', 'total_cases', 'total_deaths', 'total_tests', 'positive_rate', 
'total_vaccinations', 'people_vaccinated', 'total_boosters', 'people_fully_vaccinated']]


# Renaming the columns for better readability
df.columns = ['Country or Location', 'Cumulative Total of Confirmed Covid-19 Cases', 'Cumulative Total of Confirmed Covid-19 Deaths', 
'Total Number of Covid-19 Tests Taken', 'Share of Positive Covid-19 Tests (%)', 
'Total Number of Vaccinations Administered', 'Total Number of People who Received at Least One Vaccine Dose', 
'Total Number of People who Received Only the Booster Dose', 'Total Number of People Who Receieved All Vacine Doses']


# Replace the duplicate entries by taking the maximum value of each column for each Country or Location

df = df.groupby('Country or Location').agg({
    'Cumulative Total of Confirmed Covid-19 Cases': 'max', 
    'Cumulative Total of Confirmed Covid-19 Deaths': 'max', 
    'Total Number of Covid-19 Tests Taken': 'max', 
    'Share of Positive Covid-19 Tests (%)': 'max', 
    'Total Number of Vaccinations Administered': 'max', 
    'Total Number of People who Received at Least One Vaccine Dose': 'max',
    'Total Number of People who Received Only the Booster Dose': 'max',
    'Total Number of People Who Receieved All Vacine Doses': 'max',
    }).reset_index()


# Fill in missing values

df.fillna("Data Not Observed", inplace=True)


df.to_csv('covid19_datatable_cleaned.csv', index=False)



def round2percent(value):
    if value == "Data Not Observed":
        return value
    else:
        return round(value * 100, 2)

df['Share of Positive Covid-19 Tests (%)'] = df['Share of Positive Covid-19 Tests (%)'].apply(round2percent)


df.to_csv('covid19_datatable_cleaned.csv', index=False)


# If the entry is above 1 million, I want to round to the nearest million and then to the nearest tenth. So for
# example an entry should be 125.5 million. 

def millions_rounding(value_entry):
    if value_entry == "Data Not Observed":
        return value_entry
    value_entry = float(value_entry)
    if value_entry >= 1_000_000:
        return f"{value_entry / 1_000_000:.1f} million".replace(".0 million", " million")
    else:
        return f"{value_entry:,.0f}"




# Leave out the Positive Tests column since the data in that column is in percents

columns_to_change = [
    'Cumulative Total of Confirmed Covid-19 Cases', 
    'Cumulative Total of Confirmed Covid-19 Deaths', 
    'Total Number of Covid-19 Tests Taken', 
    'Total Number of Vaccinations Administered', 
    'Total Number of People who Received at Least One Vaccine Dose', 
    'Total Number of People who Received Only the Booster Dose',
    'Total Number of People Who Receieved All Vacine Doses'
]

for entry in columns_to_change:
    df[entry] = df[entry].apply(millions_rounding)


df.to_csv('covid19_datatable_cleaned.csv', index=False)


from sklearn.impute import KNNImputer

# We need to replace the strings with integers to perform math operations

df.replace("Data Not Observed", np.nan, inplace=True)

columns_to_use = [
    'Cumulative Total of Confirmed Covid-19 Cases', 
    'Cumulative Total of Confirmed Covid-19 Deaths', 
    'Total Number of Covid-19 Tests Taken', 
    'Total Number of Vaccinations Administered', 
    'Total Number of People who Received at Least One Vaccine Dose', 
    'Total Number of People who Received Only the Booster Dose',
    'Total Number of People Who Receieved All Vacine Doses'
]

# We need to revert back to our regular numbers we used for the dataset instead of the way we rounded to the 
# nearest million

def undo_millions_rounding(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        if "million" in value:
            return float(value.replace(" million", "")) * 1_000_000
        elif "," in value:
            return float(value.replace(",", ""))
    else:
        return value



for entry in columns_to_use:
    df[entry] = df[entry].apply(undo_millions_rounding)

# We can store our original value for the country or location column in another variable and delete
# it from the dataset temporarily to perform our algorithm 

new_countries = df['Country or Location']
dropped_column_DS = df.drop('Country or Location', axis=1)

# Now we can implement the KNN algorithm

knn_imputer = KNNImputer(n_neighbors=5) # In this example, our K value is 5, we can change this to any number
#                                         number of neighbors to run the KNN algorithm on

knn_imputeDS = knn_imputer.fit_transform(dropped_column_DS)
knn_imputeDF = pd.DataFrame(knn_imputeDS, columns=dropped_column_DS.columns)
knn_imputeDF['Country or Location'] = new_countries

# If the country or location had no neighbors to use in the KNN computation, we can fill them using the median of the columns
knn_imputeDF[columns_to_use] = knn_imputeDF[columns_to_use].fillna(knn_imputeDF[columns_to_use].median())

# Now we need to reupdate the ordering of columns and make sure our Country column is the first one

col = ['Country or Location'] + [col for col in knn_imputeDF.columns if col != 'Country or Location']
knn_imputeDF = knn_imputeDF[col]

knn_imputeDF.to_csv('KNN_covid_data_table.csv', index=False)


df = pd.read_csv('KNN_covid_data_table.csv')
df.replace("Data Not Observed", np.nan, inplace=True)

# Make sure entries in 'Share of Positive Covid-19 Tests (%)' column are rounded to the nearest hundredth
df['Share of Positive Covid-19 Tests (%)'] = df['Share of Positive Covid-19 Tests (%)'].round(2)


df.fillna("Data Not Observed", inplace=True)

# Apply the millions_rounding function to each data entry in every column 

for col in columns_to_use:
    df[col] = df[col].apply(millions_rounding)


df.to_csv('KNN_covid_data_table.csv', index=False)



import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
import os

# We use a module called os and join paths with the directory to load up our .csv files. The reason I did this 
# is so that anyone with the saved .csv files from running my code is able to perform the actions of this project.
# In other words, I did not want to load the files locally so the code is inaccessible for other people to run.

current_directory = os.getcwd()

dataset1 = pd.read_csv(os.path.join(current_directory, 'covid19_datatable_cleaned.csv'))
dataset2 = pd.read_csv(os.path.join(current_directory, 'KNN_covid_data_table.csv'))



# Now we can begin the clustering part. In this case, I used the DBSCAN algorithm to perform the dataset clustering

def updating_data_entries(df):

    # We need to drop the country or location column again since it is not numeric

    numeric_df = df.drop('Country or Location', axis=1)
    
    # We need to replace all the strings with an integer type

    numeric_df.replace("Data Not Observed", np.nan, inplace=True)

    # We have to undo the millions rounding function that we used on our data 

    for column in columns_to_use:
        numeric_df[column] = numeric_df[column].apply(undo_millions_rounding)

    # Now we can covert all data to a numeric data frame without issue

    numeric_df = numeric_df.apply(pd.to_numeric)

    # For values that are still missing we can use the median values of that respective column to fill it in

    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(numeric_df)

    # We need to make sure the data is scaled correctly and properly as we make the dataset

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    return scaled_data


# Now we can run the function on our datasets

updated_dataset1 = updating_data_entries(dataset1)
updated_dataset2 = updating_data_entries(dataset2)

# We need to define the epsilon distance (max distance between two neighboring points) and min samples
# (the minimum number of samples required for a neighbor of data entries to be considered a cluster) I used 3 and 2
# but this can be changed

eps = 3
min_samples = 2

# Finally we can correctly run the DBSCAN algorithm on our data sets using the proper formatting

DBSCAN_set1 = DBSCAN(eps=eps, min_samples=min_samples).fit(updated_dataset1)
DBSCAN_set2 = DBSCAN(eps=eps, min_samples=min_samples).fit(updated_dataset2)


# Reassess our data for cluster in each dataset

dataset1['Cluster'] = DBSCAN_set1.labels_
dataset2['Cluster'] = DBSCAN_set2.labels_




# We need to select a wide range of colors to represent the number of clusters

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
          '#f57c00', '#3f51b5', '#008080', '#006400', '#ffd700', 
          '#1e90ff', '#9932cc', '#ff1493', '#dc143c']


sorted_data = dataset1.sort_values(by=['Cumulative Total of Confirmed Covid-19 Cases', 'Cumulative Total of Confirmed Covid-19 Deaths'], ascending=True)

# Adding a new column of data for clusters

sorted_data['ClusterCount'] = sorted_data.groupby('Cluster')['Cluster'].transform('count')

# Make the scatter plot, x is number of cases, y is number of deaths

fig = px.scatter(sorted_data, 
                 x='Cumulative Total of Confirmed Covid-19 Cases', 
                 y='Cumulative Total of Confirmed Covid-19 Deaths', 
                 color='Cluster', 
                 title='COVID-19 Clusters by Country', 
                 color_discrete_sequence=colors,
                 hover_name='Country or Location')


# Fixing up and resizing the scatter plot

fig.update_layout(
    title={
        'text': "COVID-19 Clusters by Country (Dataset 1)",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title='Total Confirmed COVID-19 Cases', 
    yaxis_title='Total Confirmed COVID-19 Deaths',
    title_x=0.5)

fig.update_traces(marker_size=10, marker_opacity=0.7)
fig.write_html("covid19original_DS_cluster.html")









# We need to select a wide range of colors to represent the number of clusters

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
          '#f57c00', '#3f51b5', '#008080', '#006400', '#ffd700', 
          '#1e90ff', '#9932cc', '#ff1493', '#dc143c']

# Sort the data so it is easier to work with alter on 

sorted_data = dataset2.sort_values(by=['Cumulative Total of Confirmed Covid-19 Cases', 'Cumulative Total of Confirmed Covid-19 Deaths'], ascending=True)

# Add a column of data for the number of clusters

sorted_data['ClusterCount'] = sorted_data.groupby('Cluster')['Cluster'].transform('count')

# Make the scatter plot, x is number of cases, y is number of deaths

fig = px.scatter(sorted_data, 
                 x='Cumulative Total of Confirmed Covid-19 Cases', 
                 y='Cumulative Total of Confirmed Covid-19 Deaths', 
                 color='Cluster', 
                 title='COVID-19 Clusters by Country (Dataset 2)', 
                 color_discrete_sequence=colors,
                 hover_name='Country or Location')


# Fixing up and resizing the scatter plot

fig.update_layout(
    title={
        'text': "COVID-19 Clusters by Country (Dataset 2)",
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title='Total Confirmed COVID-19 Cases', 
    yaxis_title='Total Confirmed COVID-19 Deaths',
    title_x=0.5)

fig.update_traces(marker_size=10, marker_opacity=0.7)


fig.write_html("covid19_predictional_algorithm_cluster.html")



