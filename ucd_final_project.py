import credentials
import os
import requests
from pandas import json_normalize
import pandas as pd
import numpy as np
# For the project i am disabling SSL certificate verification, 
# however this can  create security vulnerabilities so should not be applied in production
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import re
from geopy.geocoders import GoogleV3
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle













"""
Start of Data collection 

"""

# Strava URL's 
auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"


# Strava Authentication 


res = requests.post(auth_url, data=credentials.payload, verify=False)
access_token = res.json()['access_token']
header = {'Authorization': 'Bearer ' + access_token}




request_page_num = 1

# Strava pull down a single page with 20 activites 
# This is just a tempoary solution while testing to limit the amount of data being pulled down
#param = {'per_page': 20, 'page': request_page_num}
#all_activities = requests.get(activites_url, headers=header, params=param).json()

# This is the code being used to pull all activities 
all_activities = []
while True:
    param = {'per_page': 200, 'page': request_page_num}
    my_dataset = requests.get(activites_url, headers=header, params=param).json()
    print(f' Have collected {len(my_dataset)} activities')
    if len(my_dataset) == 0:
        print("Breaking out of while loop as all activities have been collected")
        break
    if all_activities:
        print(f'all_activities has been updated and currently has {len(all_activities)} please wait')
        all_activities.extend(my_dataset)
    else:
        print(f' adding first group of activities to the dataset')
        all_activities = my_dataset

    request_page_num += 1



activities_dataset = json_normalize(all_activities)
activity_column_names = list(activities_dataset.columns.values)


# Use if issues with API this will load a backup CSV file of the last time the Strava API was run 
#activities_dataset = pd.read_csv('strava_raw_data.csv')



#Storing raw data in case its needed later 
activities_dataset.to_csv('strava_raw_data.csv', mode='w', index=False, header=True)
print("Back up of raw strava data successfully.")

"""
End of Data collection 

"""

print(activities_dataset.info())
print(activities_dataset.describe())

"""
Start of data cleaning part of the code 

"""

# convert distance in meters to kilometers
activities_dataset['distance'] = np.divide(activities_dataset['distance'], 1000)

# convert moving in seconds time to minutes
activities_dataset['moving_time'] = np.divide(activities_dataset['moving_time'], 60)

# convert elapsed time in seconds time to minutes
activities_dataset['elapsed_time'] = np.divide(activities_dataset['elapsed_time'], 60)

# convert average speed in meters per second to kilometers per hour
activities_dataset['average_speed'] = np.multiply(activities_dataset['average_speed'], 60 * 60 / 1000)

# convert max speed in meters per second to kilometers per hour
activities_dataset['max_speed'] = np.multiply(activities_dataset['max_speed'], 60 * 60 / 1000)


# Converting the date and time format
activities_dataset['date'] = activities_dataset['start_date'].apply(pd.to_datetime, format='%Y-%m-%d').dt.date
activities_dataset['month'] = activities_dataset['start_date'].apply(pd.to_datetime, format='%Y-%m-%d').dt.month
activities_dataset['start_date_local'] = pd.to_datetime(activities_dataset['start_date_local'], errors='coerce')
activities_dataset['start_time'] = activities_dataset['start_date_local'].dt.time
activities_dataset['start_time'] = activities_dataset['start_time'].astype(str)
activities_dataset['start_date'] = activities_dataset['start_date_local'].dt.date
activities_dataset['week_number'] = activities_dataset['start_date_local'].dt.strftime('%U')


# Removing small activities using numpty
mask_walk = (activities_dataset['distance'] < 1) & (activities_dataset['type'] == 'Walk')
mask_run = (activities_dataset['distance'] < 1) & (activities_dataset['type'] == 'Run')
mask_ride = (activities_dataset['distance'] < 5) & (activities_dataset['type'] == 'Ride')
mask_virtual_ride = (activities_dataset['distance'] < 5) & (activities_dataset['type'] == 'VirtualRide')

activities_dataset = activities_dataset[~np.logical_or.reduce([mask_walk, mask_run, mask_ride, mask_virtual_ride])]




# Removing any activities that might be in car
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.average_speed > 36].index) # need to review this setting to see if 36 is a good value

# Removing any activities with 0 elevation and elevation above 16000M
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.total_elevation_gain == 0].index) 
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.total_elevation_gain > 16000].index)

# Removing any activities with no heart rate
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.average_heartrate == 0].index)

# Removing any activities with heart rate above 205
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.average_heartrate >= 205].index)

# Removing any activities that are swiming
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.type == 'Swim'].index)

#Removing outliers
activities_dataset = activities_dataset.drop(activities_dataset[activities_dataset.distance > 350].index)
activities_dataset = activities_dataset.drop(activities_dataset[(activities_dataset.elapsed_time >= 1000) & (activities_dataset.type == 'Walk')].index)

# Checking dataset for duplicates and droping, using "map.id" as specific culumm to check                                            
activities_dataset = activities_dataset.drop_duplicates(subset=["map.id"])

activities_dataset = activities_dataset.reset_index(drop=True)

"""
End of data cleaning part of the code  

"""




'''
Regular Expression Code Example 

Using latitude-longitude pairs available in the start_latlng column it generates a new coloum geo_location by using the Google Maps Geocoding API
The code then looks for Eircodes within the generated geo_location strings using regular expressions. If an Eircode is found, it is extracted and inserted into a new column called eircode.
Finally, the code rewrites the dataframe with the added eircode column included and stores it as a CSV file called "strava_runs_eircode.csv".

Applying the change to the whole dataframe (activities_dataset) was taking too long using the Google API so i decided to just apply it to activities marked as "Run".

'''

# create new column location that converts start_latlng into a geo location using the Goggle API
def reverse_geocode(location):
    try:
        #removing link to credentials and hardcoding API keys for project evelulation 
        # it will be reverted once the project has been marked and API keys will also be changed
        return credentials.geolocator.reverse(location)
    except:
        return None
    

# Filter the activities dataset by selecting only 'Run' type records
run_activities = activities_dataset[activities_dataset['type'] == 'Run'].copy()

# Apply 'reverse_geocode' function on each lat-lng row and store it in geo_location column
run_activities['geo_location'] = run_activities['start_latlng'].apply(reverse_geocode)


# Converting 'geo_location' to string extracting the eircode from the address and pupulating a new field with the data 
run_activities['geo_location']= run_activities['geo_location'].astype(str)
eircode_pattern = r'\b([A-Za-z]{1}[0-9]{2}\s?[0-9AC-FHKNPRTV-Y]{4})\b'
run_activities['eircode'] = run_activities['geo_location'].apply(lambda x: re.findall(eircode_pattern, x)[0] if (re.findall(eircode_pattern, x) != []) else None)

run_geo_columns = ['week_number','start_date','start_time','type','name','geo_location','eircode','distance','moving_time',
                     'elapsed_time','average_speed','max_speed','total_elevation_gain',
                     'average_heartrate','max_heartrate','average_temp']

#rewriting the dataset to only include the inforamtion that i want
run_activities = run_activities[run_geo_columns]

# Saving the dataset as a CSV file to it can be revewed to see if colum "eircode" contains just an eircode
run_activities.to_csv('strava_data_eircode.csv', mode='w', index=False, header=True)
print("Back up of Regular Expression example dataset to strava_data_eircode.csv successfully.")



"""
Start of storage for processing later

"""
# identifying the key columes that are needed  
selecting_columns = ['week_number','start_date','start_time','type','name','distance','moving_time',
                     'elapsed_time','average_speed','max_speed','total_elevation_gain',
                     'average_heartrate','max_heartrate','average_cadence','average_temp','pr_count','average_watts',
                     'total_photo_count','achievement_count','kudos_count','comment_count',
                     'athlete_count','start_latlng','end_latlng','map.id','map.summary_polyline','date','month']

#rewriting the dataset to only include the inforamtion that i want
activities_dataset = activities_dataset[selecting_columns]

# Saving the dataset as a CSV file this will be used later when running the data analyics script 
activities_dataset.to_csv('cleaned_up_strava_data.csv', mode='w', index=False, header=True)
print("Back up of main activity dataset to strava_data.csv successfully.")

"""
Example code to show how to join 2 datasets 

"""
# Create 2 new datasets from the existing dataset spiliting the data and using "map.id" as the common coloumn
dataframe1 = activities_dataset[['week_number', 'start_date', 'start_time', 'type', 'name', 'distance', 'moving_time', 'elapsed_time', 'average_speed', 'map.id']]
dataframe2 = activities_dataset[['max_speed', 'total_elevation_gain', 'average_heartrate', 'max_heartrate', 'average_cadence', 'average_temp', 'pr_count', 'average_watts', 'total_photo_count', 'achievement_count', 'kudos_count', 'comment_count', 'athlete_count', 'start_latlng', 'end_latlng', 'map.id']]

# create a new dataset by joining 2 together 
dataframe_merged = pd.merge(dataframe1, dataframe2, on='map.id')

print(dataframe1.info())
print(dataframe2.info())
print(dataframe_merged.info())

#store the new dataset as a CSV file so that it can be checked the the two datasets have been joined 
dataframe_merged.to_csv('joined_dataset.csv', mode='w', index=False, header=True)
print("Back up of joined dataset successfully.")



''' 
This is the start of the Data Analyics part of the project 
'''


#  Loading the cleaned-up data I am just doing this to demonstrate that i can load a file as part of the project 
df = pd.read_csv('cleaned_up_strava_data.csv')

# selecting the columns that i need and producing a new dataframe 
columns_needed = ['start_date','type','moving_time', 'average_heartrate', 'max_heartrate', 'average_speed', 'distance','total_elevation_gain','max_speed','elapsed_time']
activity_data = df[columns_needed]
activity_data = activity_data.dropna().reset_index(drop=True)


activity_data['start_date'] = pd.to_datetime(activity_data['start_date'])
activity_data['year'] = activity_data['start_date'].dt.year
activity_data['month'] = activity_data['start_date'].dt.month


# Select the relevant features for clustering
features = ['moving_time', 'average_heartrate',  'average_speed', 'distance','total_elevation_gain']

activity_data_clustering = activity_data[features]


# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
activity_data_scaled = scaler.fit_transform(activity_data_clustering)

# Finding the optimal number of clusters using silhouette score
silhouette_scores = []
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(activity_data_scaled)
    silhouette_scores.append(silhouette_score(activity_data_scaled, kmeans.labels_))
# Forcing it to have 3 clusters as this is the number that i require
optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) +2

print(optimal_num_clusters)

# Perform K-means clustering
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42,n_init=10)
activity_data['Cluster'] = kmeans.fit_predict(activity_data_scaled)



# Using 3 predefined activities to indentify the cluster number as they can change when new activities are added
high_cluster = activity_data[activity_data['start_date'] == '2022-04-30']['Cluster'].values[0]
medium_cluster = activity_data[activity_data['start_date'] == '2011-08-03']['Cluster'].values[0]
low_cluster = activity_data[activity_data['start_date'] == '2020-06-19']['Cluster'].values[0]
# Adding a difficulty level name to each cluster 
difficulty_mapping = {low_cluster: 'Low Intensity', medium_cluster: 'Medium Intensity', high_cluster: 'High Intensity'}
activity_data['difficulty_label'] = activity_data['Cluster'].map(difficulty_mapping)


with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)



# Validating effort levels with K-Nearest-Neighbours KNN:
train_data_clean = activity_data[[ 'difficulty_label','moving_time', 'average_heartrate',  'average_speed', 'distance','total_elevation_gain']]
train_data_clean = train_data_clean.dropna()

test_data = activity_data[[ 'difficulty_label','moving_time', 'average_heartrate',  'average_speed', 'distance','total_elevation_gain']]
X = test_data.drop('difficulty_label', axis=1)

y = train_data_clean['difficulty_label']
target_names = ["Low Intensity", "Medium Intensity", "High Intensity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.7)
X_scaler = StandardScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Validating effort levels with K-Nearest-Neighbours KNN:
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
knn_accuracy_score = knn.score(X_test_scaled, y_test)
print('K-Nearest-Neighbours Accuracy: {:.2%}'.format(knn_accuracy_score))
predictions = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)
disp.plot()
plt.title('K-Nearest-Neighbours KNN')
plt.show()

# storing KNN model  
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)



# Validating effort levels with Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_accuracy_score = rf.score(X_test_scaled, y_test)
print('Random Forest Accuracy: {:.2%}'.format(rf_accuracy_score))
predictions = rf.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()
plt.title('Random Forest Classifier')
plt.show()

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf, file)


# Validating effort levels with Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_accuracy_score = dt.score(X_test_scaled, y_test)
print('Decision Tree Accuracy: {:.2%}'.format(dt_accuracy_score))
predictions = dt.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot()
plt.title('Decision Tree Classifier')
plt.show()

with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt, file)


# Create the boxplot with jitter
color_palette = {'High Intensity': 'red', 'Medium Intensity': 'orange', 'Low Intensity': 'green'}
sns.boxplot(data=activity_data, x='year', y='distance')
sns.stripplot(data=activity_data, x='year', y='distance', palette=color_palette, hue='difficulty_label', jitter=True)
plt.title('Activity Difficulty Level per Year')
plt.xlabel('Year')
plt.ylabel('Distance KM')
plt.show()


# Producing a lineplot graph of the cumulative distance per year
df_years = (activity_data[activity_data['start_date'].dt.year.isin(range(2011, 2024))]
             .sort_values(by='start_date')
             .assign(cumulative_distance=lambda d: d.groupby(d['start_date'].dt.year)['distance'].cumsum())
            )

for year,  in zip(range(2011,2024)):
    mask = (df_years['start_date'].dt.year == year)
    plt.plot(df_years.loc[mask, 'start_date'].dt.day_of_year,
             df_years.loc[mask, 'cumulative_distance'],
              label=str(year))

plt.title('Cumulative Distance Traveled in 2011 - 2024')
plt.xlabel('Days of year')
plt.ylabel('Distance KM')
plt.legend()
plt.show()



fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for row in range(0, 3):
    for col in range(0, 3):
        # Calculate the year based on the current row and column
        year = 2015 + row * 3 + col

        # Filter the data for the current year
        filtered_data = activity_data[activity_data['year'] == year]

        # Group the filtered data by year, month, and difficulty_label, and calculate the sum of distance
        grouped_data = filtered_data.groupby(['year', 'month', 'difficulty_label'])['distance'].sum().reset_index()

        # Plot the total distances per month per year per difficulty_label on the current subplot
        sns.barplot(data=grouped_data, x='month', y='distance',
                    palette=color_palette, hue='difficulty_label', ax=axes[row, col])

        # Set the title of each subplot to show the year
        axes[row, col].set_title(str(year))

# Adjust the spacing between subplots
plt.tight_layout()

# Show the grid of subplots
plt.show()

 
# Save the dataframe with the cluster labels
activity_data.to_csv('data_with_cluster_info.csv', mode='w', index=False, header=True)

print(' Everything is finished now')