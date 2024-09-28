import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the Zomato dataset
zomato_df = pd.read_csv('zomato_df_final_data.csv')

# Load the Sydney GeoJSON data
sydney_geo = gpd.read_file('sydney.geojson')

# --- PART A: 1. Provide plots/graphs to support the following questions ---

# Display basic information about the Zomato DataFrame (Data exploration)
print("Zomato DataFrame Info:")
zomato_df.info()  # Shows data types and missing values

# Display first five rows of the Zomato dataset (Data preview)
print("\nFirst 5 rows of Zomato Data:")
print(zomato_df.head())

# --- 1.1 How many unique cuisines are served by Sydney restaurants? ---
# Convert 'cuisine' column from a string representation to an actual list
zomato_df.loc[:, 'cuisine'] = zomato_df['cuisine'].apply(lambda x: eval(x))

# Explode the list of cuisines into individual entries
exploded_cuisines = zomato_df.explode('cuisine')

# Count unique cuisines
unique_cuisines_count = exploded_cuisines['cuisine'].nunique()
print(f"Number of unique cuisines: {unique_cuisines_count}")

# Plot the top 10 most common cuisines
top_cuisines = exploded_cuisines['cuisine'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_cuisines.plot(kind='bar', color='lightblue')
plt.title("Top 10 Most Common Cuisines in Sydney")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()

# --- 1.2 Which suburbs (top 3) have the highest number of restaurants? ---
# Count the number of restaurants in each suburb (subzone)
top_suburbs = zomato_df['subzone'].value_counts().head(3)
print("Top 3 suburbs with the most restaurants:")
print(top_suburbs)

# --- 1.3 Analyze the relationship between cost and rating ---
# Define the custom order for the ratings
rating_order = ["Excellent", "Very Good", "Good", "Average", "Poor"]

# Convert 'rating_text' to a categorical type with the specific order
zomato_df.loc[:, 'rating_text'] = pd.Categorical(zomato_df['rating_text'], categories=rating_order, ordered=True)

# Plot a boxplot to analyze the relationship between cost and rating
plt.figure(figsize=(10, 6))
sns.boxplot(data=zomato_df, x='rating_text', y='cost', order=rating_order)
plt.title("Cost Distribution by Restaurant Rating")
plt.xlabel("Restaurant Rating")
plt.ylabel("Cost for Two People (AUD)")
plt.xticks(rotation=45)
plt.show()

# --- PART A: 2. Perform exploratory analysis for the following variables ---

# --- 2.1 Exploratory analysis for 'Cost' ---
plt.figure(figsize=(10, 6))
sns.histplot(zomato_df['cost'], bins=20, kde=True, color='skyblue')
plt.title("Cost Distribution of Restaurants")
plt.xlabel("Cost for Two People (AUD)")
plt.ylabel("Frequency")
plt.show()

# --- 2.2 Exploratory analysis for 'Rating' ---
plt.figure(figsize=(10, 6))
sns.histplot(zomato_df['rating_number'], bins=10, kde=True, color='lightgreen')
plt.title("Rating Distribution of Restaurants")
plt.xlabel("Rating (Out of 5)")
plt.ylabel("Frequency")
plt.show()

# --- 2.3 Exploratory analysis for 'Type' ---

# Get the count of each restaurant type
restaurant_type_counts = zomato_df['type'].value_counts()

# Keep the top 20 restaurant types, and sum the rest into 'Others'
top_20_types = restaurant_type_counts.head(20)
others_count = restaurant_type_counts[20:].sum()

# Add the 'Others' category
top_20_types['Others'] = others_count

# Plot the modified bar chart for the top 20 types + 'Others'
plt.figure(figsize=(10, 6))
top_20_types.plot(kind='bar', color='coral')
plt.title("Distribution of Restaurant Types (Top 20 + Others)")
plt.xlabel("Restaurant Type")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()

# --- PART A: 3. Cuisine Density Map ---

# Clean and prepare data for visualization
cuisine_density = zomato_df.groupby('subzone')['cuisine'].count().reset_index()
cuisine_density.columns = ['subzone', 'restaurant_count']

# Merge with geojson data
geo_data_merged = sydney_geo.merge(cuisine_density, left_on='SSC_NAME', right_on='subzone', how='left')
geo_data_merged['restaurant_count'] = geo_data_merged['restaurant_count'].fillna(0)

# Plot cuisine density map using geopandas
plt.figure(figsize=(12, 8))
geo_data_merged.plot(column='restaurant_count', cmap='OrRd', legend=True, 
                     legend_kwds={'label': "Number of Restaurants", 'orientation': "horizontal"})
plt.title("Cuisine Density by Suburb in Sydney")
plt.show()

# --- PART A: 4. Interactive Plotly visualizations ---

# --- Interactive bar chart of restaurant types ---
top_types = zomato_df['type'].value_counts().head(10).reset_index()
top_types.columns = ['Restaurant Type', 'Count']

fig = px.bar(top_types, x='Restaurant Type', y='Count',
             labels={'Restaurant Type': 'Restaurant Type', 'Count': 'Number of Restaurants'},
             title="Top 10 Most Common Restaurant Types")
fig.show()

# --- Interactive scatter map using Plotly and Mapbox ---
px.set_mapbox_access_token('pk.eyJ1IjoieWVleWV1bmciLCJhIjoiY20xaXpkd3piMDAwNjJqb2xzMjllYXhxMyJ9.OEx8wP0gLWMreHCvoADdnQ')

# Drop rows where 'lat', 'lng', or 'cost' are NaN for plotting
zomato_df_clean = zomato_df.dropna(subset=['lat', 'lng', 'cost'])

fig = px.scatter_mapbox(zomato_df_clean, lat="lat", lon="lng", color="subzone", size="cost",
                        hover_name="title", hover_data=["cuisine", "rating_text"],
                        title="Interactive Cuisine Density Map in Sydney", zoom=10)
fig.show()

# --- PART A: 5. Export cleaned data for Tableau ---

zomato_df.to_csv('zomato_cleaned_data_for_tableau.csv', index=False)

#Tableau Dashboard Link
#https://public.tableau.com/views/Zomato_Sydney_Data_Analysis/ZomatoSydneyrestaurants?:language=zh-CN&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
