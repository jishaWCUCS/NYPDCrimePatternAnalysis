#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os 
from matplotlib import pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.decomposition import PCA

crimeDataNYPD = pd.read_csv("NYPDArrestsDataHistoric.csv", dtype={'PD_CD':str, 'KY_CD':str, 'JURISDICTION_CODE':str})
crimeDataNYPD.info()


# In[2]:


pd.set_option('display.max_rows', 50000000)


# In[3]:


# Displaying the first few rows of the dataset
crimeDataNYPD.head()


# In[4]:


# Basic Info about the Dataset
# Shape of the dataset
crimeDataNYPD.shape


# In[5]:


# Data types of the columns
crimeDataNYPD.dtypes


# In[6]:


# Summary statistics
crimeDataNYPD.describe(include='all')


# In[7]:


crimeDataNYPD.isnull().any()


# In[8]:


# Identify missing values
missing_values = crimeDataNYPD.isnull().sum()
print(missing_values[missing_values > 0])


# In[9]:


missing_percentage = crimeDataNYPD.isnull().sum() * 100 / len(crimeDataNYPD)
print(missing_percentage) 


# In[10]:


crimeDataNYPD.dropna(subset=['PD_CD','LAW_CODE', 'AGE_GROUP', 'JURISDICTION_CODE', 'ARREST_BORO', 'X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'],inplace=True)


# In[11]:


# Convert the 'ARREST_DATE' column to a datetime object
crimeDataNYPD['ARREST_DATE'] = pd.to_datetime(crimeDataNYPD['ARREST_DATE'], format='%m/%d/%Y')


# In[12]:


# Replace any inconsistent labels with the correct ones
crimeDataNYPD['LAW_CAT_CD'] = crimeDataNYPD['LAW_CAT_CD'].replace({'felony': 'F', 'misdemeanor': 'M'})
crimeDataNYPD['ARREST_BORO'] = crimeDataNYPD['ARREST_BORO'].replace({'Q': 'Queens', 'M': 'Manhattan', 
                                                                     'S': 'Staten Island', 'B': 'Bronx', 'K': 'Brooklyn'})
crimeDataNYPD['PERP_SEX'] = crimeDataNYPD['PERP_SEX'].replace({'M':'Male', 'F':'Female'})
crimeDataNYPD['OFNS_DESC'] = crimeDataNYPD['OFNS_DESC'].replace({'HARRASSMENT 2':'HARRASSMENT','ESCAPE 3':'ESCAPE',
                                                                 'CHILD ABANDONMENT/NON SUPPORT 1':'CHILD ABANDONMENT/NON SUPPORT'})
# Check for unique values to find inconsistencies
print(crimeDataNYPD['AGE_GROUP'].unique())
print(crimeDataNYPD['OFNS_DESC'].unique())


# In[13]:


# First, let's define a function to categorize the age data
def categorize_age(age):
    if age in ['<18', '18-24', '25-44', '45-64', '65+']:
        return age
    elif age == 'UNKNOWN':
        return 'UNKNOWN'
    else:
        # Assuming the year of birth is provided instead of the age group
        # You would need to adjust the reference year if necessary
        birth_year = int(age)
        reference_year = 2017  # Assuming data is from 2017 as per the date on the file
        age = reference_year - birth_year
        if age < 18:
            return '<18'
        elif age <= 24:
            return '18-24'
        elif age <= 44:
            return '25-44'
        elif age <= 64:
            return '45-64'
        else:
            return '65+'
        
# Apply the function to the AGE_GROUP column
crimeDataNYPD['AGE_GROUP'] = crimeDataNYPD['AGE_GROUP'].apply(categorize_age)

# Now check the unique values to confirm they're cleaned
print(crimeDataNYPD['AGE_GROUP'].unique())


# In[14]:


# Verify latitude and longitude ranges for NYC
crimeDataNYPD = crimeDataNYPD[(crimeDataNYPD['Latitude'] >= 40.477399) & (crimeDataNYPD['Latitude'] <= 40.917577) &
            (crimeDataNYPD['Longitude'] >= -74.25909) & (crimeDataNYPD['Longitude'] <= -73.700009)]
# Check the number of records after filtering
print(crimeDataNYPD.shape)

# Display the first few records to confirm the filter
print(crimeDataNYPD.head())


# In[15]:


crimeDataNYPD['PD_DESC'] = crimeDataNYPD['PD_DESC'].str.upper().str.strip()
crimeDataNYPD['OFNS_DESC'] = crimeDataNYPD['OFNS_DESC'].str.upper().str.strip()

print(crimeDataNYPD['OFNS_DESC'].unique())


# In[16]:


# Extract year and month from the ARREST_DATE for further analysis
crimeDataNYPD['YEAR'] = crimeDataNYPD['ARREST_DATE'].dt.year
crimeDataNYPD['MONTH'] = crimeDataNYPD['ARREST_DATE'].dt.month

# You can also extract day of the week, etc.
crimeDataNYPD['DAY_OF_WEEK'] = crimeDataNYPD['ARREST_DATE'].dt.day_name()
print(crimeDataNYPD['ARREST_DATE'].unique())
print(crimeDataNYPD['DAY_OF_WEEK'].unique())


# In[17]:


crimeDataNYPD.isnull().sum()


# In[18]:


crimeDataNYPD.isna().sum()


# In[19]:


def infer_law_cat(ky_cd):
    # Placeholder for the actual logic you'd use to infer the law category
    return 'Inferred Category'

# Apply the function to fill missing LAW_CAT_CD based on KY_CD
crimeDataNYPD.loc[crimeDataNYPD['LAW_CAT_CD'].isnull(), 'LAW_CAT_CD'] = crimeDataNYPD['KY_CD'].apply(infer_law_cat)
# Check missing values again after handling
print(crimeDataNYPD.isnull().sum())
print(crimeDataNYPD['PD_CD'].unique())
print(crimeDataNYPD['PD_DESC'].unique())
print(crimeDataNYPD['JURISDICTION_CODE'].unique())


# In[20]:


def infer_law_cat(ky_cd):
    # Placeholder for the actual logic you'd use to infer the law category
    return 'Inferred Category'

# Apply the function to fill missing LAW_CAT_CD based on KY_CD
crimeDataNYPD.loc[crimeDataNYPD['LAW_CAT_CD'].isnull(), 'LAW_CAT_CD'] = crimeDataNYPD['KY_CD'].apply(infer_law_cat)

# Check missing values again after handling
print(crimeDataNYPD.isnull().sum())


# In[21]:


print(crimeDataNYPD['LAW_CAT_CD'].unique())
print(crimeDataNYPD['PD_DESC'].unique())


# In[22]:


crimeDataNYPD['PD_DESC'] = crimeDataNYPD['PD_DESC'].replace({'PROSTITUTION 4':'PROSTITUTION'})

crimeDataNYPD['PD_DESC'] = crimeDataNYPD['PD_DESC'].replace({'DRUG PARAPHERNALIA,   POSSESSES OR SELLS 1':'DRUG PARAPHERNALIA,   POSSESSES OR SELLS'})
crimeDataNYPD['PD_DESC'] = crimeDataNYPD['PD_DESC'].replace({'DRUG PARAPHERNALIA,   POSSESSES OR SELLS 2':'DRUG PARAPHERNALIA,   POSSESSES OR SELLS'})


# In[23]:


def map_to_desc(code):
    mapping_dict = {}  # You would fill this with your actual mapping
    return mapping_dict.get(code, 'Unknown')

# Apply the mapping function to the missing values
crimeDataNYPD.loc[crimeDataNYPD['PD_DESC'].isnull(), 'PD_DESC'] = crimeDataNYPD['PD_CD'].apply(map_to_desc)
crimeDataNYPD.isna().sum()
# Identify missing values
missing_values = crimeDataNYPD.isnull().sum()
print(missing_values[missing_values > 0])
print(crimeDataNYPD['PD_DESC'].unique())


# In[24]:


def map_to_desc(code):
    mapping_dict = {}  # You would fill this with your actual mapping
    return mapping_dict.get(code, 'Unknown')

# Apply the mapping function to the missing values
crimeDataNYPD.loc[crimeDataNYPD['PD_DESC'].isnull(), 'PD_DESC'] = crimeDataNYPD['PD_CD'].apply(map_to_desc)


# In[25]:


print(crimeDataNYPD.isnull().sum())
crimeDataNYPD.info()


# In[26]:


crimeDataNYPD.head()


# In[27]:


# Display unique values for categorical columns
for col in crimeDataNYPD.select_dtypes('object').columns:
    print(f"{col}: {crimeDataNYPD[col].unique()}")


# In[28]:


crimeDataNYPD = crimeDataNYPD.rename(columns={'ARREST_DATE':'Arrest Date', 'ARREST_KEY':'ID', 'PD_CD':'PDcode', 'PD_DESC':'PDdesc', 'KY_CD':'KYdesc', 'OFNS_DESC':'OFNS_DESC', 'LAW_CODE':'Law Code', 'LAW_CAT_CD':'Law category', 'ARREST_BORO':'BORO_NM', 'ARREST_PRECINCT':'Precinct', 'JURISDICTION_CODE':'Jurisdiction', 'AGE_GROUP':'Susp_Age', 'PERP_SEX':'Susp_Sex', 'PERP_RACE':'Susp_Race'}, index={'ARREST_DATE': 'Date'})


# In[29]:


# Identify missing values
missing_values = crimeDataNYPD.isnull().sum()
print(missing_values[missing_values > 0])


# In[30]:


# Drop rows where 'PD_DESC', 'KY_CD', or 'OFNS_DESC' is missing
data_cleaned = crimeDataNYPD.dropna(subset=['KYdesc', 'OFNS_DESC'])

# Verify the rows have been dropped
missing_values_cleaned = data_cleaned.isnull().sum()
print(missing_values_cleaned)


# In[31]:


# Drop rows where 'PD_DESC', 'KY_CD', or 'OFNS_DESC' is missing
data_cleaned = crimeDataNYPD.dropna(subset=['KYdesc', 'OFNS_DESC'])

# Check the result
missing_values_cleaned = data_cleaned.isnull().sum()
print(missing_values_cleaned[missing_values_cleaned > 0])
crimeDataNYPD.info()


# In[32]:


crimeDataNYPD['BORO_NM'] = crimeDataNYPD['BORO_NM'].replace({'Q': 'Queens', 'M': 'Manhattan', 'S': 'Staten Island', 'B': 'Bronx', 'K': 'Brooklyn'})
crimeDataNYPD['Susp_Sex'] = crimeDataNYPD['Susp_Sex'].replace({'M': 'Male', 'F': 'Female'})

crimeDataNYPD['Law category'] = crimeDataNYPD['Law category'].replace({'F': 'Felony', 'M': 'Misdemeanor', 'V': 'Violation'})
crimeDataNYPD['Jurisdiction'].unique()
crimeDataNYPD['Jurisdiction'] = crimeDataNYPD['Jurisdiction'].replace({'0': 'Patrol', '1': 'Transit', '2': 'Housing'})


# In[33]:


crimeDataNYPD.dtypes


# In[34]:


# Data conversion
# Convert categorical data to 'category' data type
categorical_columns = ['OFNS_DESC', 'Law category']
for col in categorical_columns:
    crimeDataNYPD[col] = crimeDataNYPD[col].astype('category')


# In[35]:


# Display the first few records to confirm the filter
print(crimeDataNYPD.head())


# In[36]:


# Check for and remove duplicate rows
duplicates = crimeDataNYPD.duplicated()
print("Number of duplicate rows:", duplicates.sum())
crimeDataNYPD = crimeDataNYPD.drop_duplicates()


# In[37]:


#separating numerical and string data
numData = crimeDataNYPD.select_dtypes('number').columns
stringData = crimeDataNYPD.select_dtypes('object').columns
print(f'numData Columns:  {crimeDataNYPD[numData].columns}')
print('\n')
print(f'stringData Columns: {crimeDataNYPD[stringData].columns}')


# In[38]:


# Encoding categorical variables
label_encoder = LabelEncoder()
crimeDataNYPD['PDdesc'] = label_encoder.fit_transform(crimeDataNYPD['PDdesc'])
crimeDataNYPD['Law Code'] = label_encoder.fit_transform(crimeDataNYPD['Law Code'])
crimeDataNYPD['KYdesc'] = label_encoder.fit_transform(crimeDataNYPD['KYdesc'])


# In[39]:


#separating numerical and string data
numData = crimeDataNYPD.select_dtypes('number').columns
stringData = crimeDataNYPD.select_dtypes('object').columns
print(f'numData Columns:  {crimeDataNYPD[numData].columns}')
print('\n')
print(f'stringData Columns: {crimeDataNYPD[stringData].columns}')
crimeDataNYPD.head()


# In[40]:


newCrimeDataNYPD = crimeDataNYPD


# In[41]:


newCrimeDataNYPD.shape


# In[42]:


newCrimeDataNYPD.info()


# In[43]:


newCrimeDataNYPD.tail()


# In[44]:


newCrimeDataNYPD.isnull().sum()


# In[45]:


# Check if any values could not be converted
if newCrimeDataNYPD['Longitude'].isnull().any():
    print("Non-numeric values found in 'Longitude'")
if newCrimeDataNYPD['Latitude'].isnull().any():
    print("Non-numeric values found in 'Latitude'")


# In[46]:


# List of columns to check for outliers
columns_to_check = ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude']

# Dictionary to store outliers information
outliers_info = {}

for column in columns_to_check:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
    Q1 = newCrimeDataNYPD[column].quantile(0.25)
    Q3 = newCrimeDataNYPD[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Determine the outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    outliers = newCrimeDataNYPD[(newCrimeDataNYPD[column] < lower_bound) | (newCrimeDataNYPD[column] > upper_bound)]
    
    # Store information about outliers
    outliers_info[column] = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'number_of_outliers': outliers.shape[0]
    }

# Print the outliers information
for column, info in outliers_info.items():
    print(f"{column} - Outliers lower than {info['lower_bound']}: {info['number_of_outliers']}")
    print(f"{column} - Outliers higher than {info['upper_bound']}: {info['number_of_outliers']}")


# In[47]:


def remove_outliers(df, column):
    Q1 = newCrimeDataNYPD[column].quantile(0.25)
    Q3 = newCrimeDataNYPD[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Columns to clean
columns_to_clean = ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude']

# Apply the outlier removal function to each column
for column in columns_to_clean:
    newCrimeDataNYPD = remove_outliers(newCrimeDataNYPD, column)
newCrimeDataNYPD[columns_to_clean].plot(kind='box', subplots=True, layout=(1, len(columns_to_clean)), figsize=(12, 6))
# Print the outliers information
for column, info in outliers_info.items():
    print(f"{column} - Outliers lower than {info['lower_bound']}: {info['number_of_outliers']}")
    print(f"{column} - Outliers higher than {info['upper_bound']}: {info['number_of_outliers']}")
plt.show()


# In[48]:


# Check the number of records after filtering
crimeDataNYPD.head()
crimeDataNYPD.tail()
newCrimeDataNYPD.head(10)


# In[49]:


# Select only numeric columns for correlation analysis
numeric_df = crimeDataNYPD.select_dtypes(include=['number'])
# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
newCrimeDataNYPD.head(10)


# In[50]:


newCrimeDataNYPD['BORO_NM'].value_counts().head(15).plot.pie(radius=2.1, autopct='%1.1f%%', textprops=dict(color="black"))


# In[51]:


# Count the occurrences of each crime type (Crm Cd Desc)
# Visualize the top 20 crimes
top20_crimes = newCrimeDataNYPD['OFNS_DESC'].value_counts().head(20)
plt.figure(figsize=(10, 8))
top20_crimes.plot(kind='bar',color='orange')
plt.title('Top 20 Crimes')
plt.ylabel('Frequency')
plt.xlabel('Offense Description')
plt.show()
newCrimeDataNYPD.head(10)


# In[52]:


# Creating a cross-tabulation of the frequency of each crime type (OFNS_DESC) in each borough (BORO_NM)
crime_borough_crosstab = pd.crosstab(newCrimeDataNYPD['OFNS_DESC'], newCrimeDataNYPD['BORO_NM'])

# Using a heatmap to visualize the correlation
plt.figure(figsize=(10, 8))
sns.heatmap(crime_borough_crosstab, cmap="YlGnBu", annot=False, cbar=True)
plt.title('Frequency of Crime Types by Borough')
plt.xlabel('Borough')
plt.ylabel('Type of Crime')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# In[53]:


newCrimeDataNYPD['Susp_Age'].value_counts().head(15).plot.pie(radius=2.7, autopct='%1.1f%%', textprops=dict(color="black"))


# In[54]:


# Distribution of crimes by borough
susp_Race = newCrimeDataNYPD['Susp_Race'].value_counts()
plt.figure(figsize=(10, 6))
susp_Race.plot(kind='bar', color='red')
plt.title('Crimes by Race')
plt.ylabel('Frequency')
plt.xlabel('Race')
plt.show()


# In[55]:


#Data Visualization
# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Histograms for 'Latitude' and 'Longitude'
plt.figure(figsize=(14, 6))

# Histogram for 'Latitude'
plt.subplot(1, 2, 1)
sns.histplot(newCrimeDataNYPD['Latitude'].dropna(), bins=50, kde=False)
plt.title('Distribution of Incident Latitudes')


# In[56]:


# Dropping rows with NaN or infinite values in 'Longitude' and 'Latitude'
newCrimeDataNYPD = newCrimeDataNYPD.dropna(subset=['Longitude', 'Latitude'])
newCrimeDataNYPD = newCrimeDataNYPD[(newCrimeDataNYPD['Longitude'].notnull()) & (newCrimeDataNYPD['Latitude'].notnull())]

# Data Sampling for Plotting
#Data Visualization
sampled_data = newCrimeDataNYPD.sample(1000) if len(newCrimeDataNYPD) > 1000 else newCrimeDataNYPD

# Plotting using Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(sampled_data['Longitude'], sampled_data['Latitude'])
plt.title('Geospatial Distribution of Crimes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
newCrimeDataNYPD.head(10)


# In[57]:


#Clustering
#Selecting relevant features
print(newCrimeDataNYPD.columns)
features = newCrimeDataNYPD[['Latitude', 'Longitude']]

# Handling missing values: Option 1 - dropping rows with missing values
features_cleaned = features.dropna()

# Standardization (if necessary)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_cleaned)

# Elbow Method to choose the number of clusters
inertia = []
for i in range(1, 11):  # testing 1 to 10 clusters
    kmeans = KMeans(n_init=10,n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[58]:


newCrimeDataNYPD.head(10)
# Apply K-means clustering
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add the cluster ID to the sampled data
newCrimeDataNYPD['Cluster'] = clusters

# Visualization of clusters using a scatter plot
plt.scatter(newCrimeDataNYPD['Longitude'], newCrimeDataNYPD['Latitude'], c=newCrimeDataNYPD['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Geographic Distribution of Crime Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
print(sampled_data[['Latitude', 'Longitude']].dtypes)
newCrimeDataNYPD.head(10)


# In[59]:


newCrimeDataNYPD.head(10)
# Convert the 'Arrest Date' column to datetime format and set as index
newCrimeDataNYPD.set_index('Arrest Date', inplace=True)

# Resampling to get monthly crime counts
monthly_crimes = newCrimeDataNYPD.resample('M').size()

# Plotting monthly crime counts
plt.figure(figsize=(12, 6))
monthly_crimes.plot()
plt.title('Monthly Crime Counts')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.show()


# In[60]:


# Seasonality Analysis: Average crime counts by month
# Extract month from the index
newCrimeDataNYPD['MONTH'] = newCrimeDataNYPD.index.month

# Group by month and calculate the average
monthly_avg_crimes = newCrimeDataNYPD.groupby('MONTH').size().groupby('MONTH').mean()

# Plotting average monthly crime counts
plt.figure(figsize=(12, 6))
monthly_avg_crimes.plot(kind='bar')
plt.title('Average Monthly Crime Counts')
plt.xlabel('MONTH')
plt.ylabel('Average Number of Crimes')
plt.xticks(ticks=range(12), labels=[f'Month {i+1}' for i in range(12)], rotation=45)
plt.show()


# In[61]:


# Day of the Week Analysis: Calculating the total number of arrests for each day of the week
arrests_per_day_of_week = newCrimeDataNYPD['DAY_OF_WEEK'].value_counts()

# Plotting the distribution of arrests across days of the week
plt.figure(figsize=(8, 6))
arrests_per_day_of_week.plot(kind='bar', color='lightgreen')
plt.title('Arrests Distribution by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Arrests')
plt.xticks(rotation=45)
plt.show()


# In[62]:


# Day of the Week Analysis: Calculating the total number of arrests for each day of the week
arrests_per_day_of_week = newCrimeDataNYPD['DAY_OF_WEEK'].value_counts()
# Sorting the days of the week in order
sorter = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
arrests_per_day_of_week = arrests_per_day_of_week.reindex(sorter)

# Plotting the distribution of arrests across days of the week
plt.figure(figsize=(8, 6))
arrests_per_day_of_week.plot(kind='bar', color='green')
plt.title('Arrests Distribution by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Arrests')

plt.show()


# In[63]:


newCrimeDataNYPD.head(10)
newCrimeDataNYPD.tail(10)

if newCrimeDataNYPD[['Latitude', 'Longitude']].isnull().values.any():
    print("Data contains NaN values.")
if np.isinf(newCrimeDataNYPD[['Latitude', 'Longitude']].values).any():
    print("Data contains infinite values.")


# In[64]:


# Drop any rows with NaNs that resulted from the conversion
newCrimeDataNYPD = newCrimeDataNYPD.dropna(subset=['Latitude', 'Longitude'])


# In[65]:


newCrimeDataNYPD.head(10)


# In[66]:


newCrimeDataNYPD.dtypes


# In[67]:


newCrimeDataNYPD.info()


# In[68]:


label_encoder = LabelEncoder()
newCrimeDataNYPD['BORO_NM'] = label_encoder.fit_transform(newCrimeDataNYPD['BORO_NM'])
newCrimeDataNYPD['OFNS_DESC'] = label_encoder.fit_transform(newCrimeDataNYPD['OFNS_DESC'])
newCrimeDataNYPD.info()
newCrimeDataNYPD = newCrimeDataNYPD.dropna(subset=['DAY_OF_WEEK', 'MONTH','Precinct','Jurisdiction'])
newCrimeDataNYPD.info()


# In[69]:


newCrimeDataNYPD.shape


# In[70]:


# Assuming 'OFNS_DESC' is your target variable and newCrimeDataNYPD is your DataFrame
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(newCrimeDataNYPD['OFNS_DESC'])

# Compute class weights for all classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = dict(enumerate(weights))


# In[71]:


# Sampling a subset of the data to reduce memory usage
sampled_data = newCrimeDataNYPD.sample(frac=0.7, random_state=42)  # Taking 70% of the data
# Selecting features and target variable from the sampled data
X_sampled = sampled_data[['BORO_NM']]  # Note the double brackets to keep it in DataFrame format
y_sampled = sampled_data['OFNS_DESC']

# Splitting the sampled dataset into training and test sets
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(
    X_sampled, y_sampled, test_size=0.3, random_state=42
)
# Defining the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['BORO_NM'])
    ]
)
# Creating a pipeline for the RandomForestClassifier
pipeline_sampled = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
])
# Training the RandomForest model on the sampled data
pipeline_sampled.fit(X_train_sampled, y_train_sampled)
# Making predictions on the test set and evaluating the model
y_pred_sampled = pipeline_sampled.predict(X_test_sampled)
classification_report_sampled = classification_report(y_test_sampled, y_pred_sampled)

print(classification_report_sampled)


# In[72]:


newCrimeDataNYPD.dtypes


# In[73]:


newCrimeDataNYPD.tail()


# In[74]:


# Filter and convert types if necessary
hm_pol = newCrimeDataNYPD[newCrimeDataNYPD['Jurisdiction'] == 'Transit']
if hm_pol['Latitude'].dtype != 'float':
    hm_pol['Latitude'] = hm_pol['Latitude'].astype(float)
if hm_pol['Longitude'].dtype != 'float':
    hm_pol['Longitude'] = hm_pol['Longitude'].astype(float)

# Remove NaN values
hm_pol = hm_pol.dropna(subset=['Latitude', 'Longitude'])

# Create a map centered around an average location of the filtered data
m = folium.Map(location=[hm_pol['Latitude'].mean(), hm_pol['Longitude'].mean()], zoom_start=11)
# Create heat map data using list comprehension
heat_data = [[lat, lon] for lat, lon in zip(hm_pol['Latitude'], hm_pol['Longitude'])]
# Add HeatMap to the map
HeatMap(heat_data).add_to(m)

# Display the map
m


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




