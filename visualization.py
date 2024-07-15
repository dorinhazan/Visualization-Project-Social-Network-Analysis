import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="App Usage Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the CSV file into a DataFrame
file_path = 'cleaned_survey_data.csv'
df = pd.read_csv(file_path)

# Binary replacement dictionary
binary = {
    'Using': 1,
    'Not Using': 0,
}

# Apply the replacement to the selected columns
columns_to_rename = [col for col in df.columns if col.startswith('B-')]
for col in columns_to_rename:
    df[col] = df[col].replace(binary)

# Streamlit app
st.title('App Usage Analysis')

# Select category for analysis
category = st.selectbox('Select a Category', ['Age_group', 'Gender', 'Income', 'App Usage Frequency', 'Religion', 'Area'])

# Extract relevant columns for the selected category
if category == 'Age_group':
    category_order = ['18-24', '25-30', '31-39', '40-49', '50-64', '65+']
    df[category] = pd.Categorical(df[category], categories=category_order, ordered=True)
elif category == 'App Usage Frequency':
    frequency_columns = [col for col in df.columns if col.startswith('Frequency-')]
    df_long_freq = pd.melt(df, id_vars=['Gender', 'Income', 'Religion', 'Area'], value_vars=frequency_columns,
                           var_name='AppName', value_name='Frequency')
    df_long_freq['AppName'] = df_long_freq['AppName'].str.replace('Frequency-', '')
    main_apps = ['Facebook', 'YouTube', 'Instagram', 'TikTok', 'Twitter', 'LinkedIn', 'WhatsApp']
    df_long_freq = df_long_freq[df_long_freq['AppName'].isin(main_apps)]
    df_long_freq = df_long_freq.rename(columns={'Frequency': category})
    df = df_long_freq
    category_order = ['Not Using', 'Less often', 'Several times a month', 'Several times a week', 'Daily']
    df[category] = pd.Categorical(df[category], categories=category_order, ordered=True)
elif category == 'Religion' or category == 'Area':
    category_order = df[category].unique()
else:
    category_order = df[category].unique()

# Transform the dataset to a long format
if category != 'App Usage Frequency':
    df_long = pd.melt(df, id_vars=[category], value_vars=columns_to_rename,
                      var_name='AppName', value_name='Usage')
    df_long = df_long[df_long['Usage'] == 1]  # Filter only those who are using the app
else:
    df_long = df.copy()

# Clean the 'AppName' column
df_long['AppName'] = df_long['AppName'].str.replace('B-', '')

# Filter to include only main applications
main_apps = ['Facebook', 'YouTube', 'Instagram', 'TikTok', 'Twitter', 'LinkedIn', 'WhatsApp']
df_long = df_long[df_long['AppName'].isin(main_apps)]

# Calculate the count of users for each app and category
if category != 'App Usage Frequency':
    df_count = df_long.groupby(['AppName', category]).size().reset_index(name='UserCount')
    df_total = df_long.groupby('AppName').size().reset_index(name='TotalUsers')
else:
    df_count = df_long.groupby([category, 'AppName']).size().reset_index(name='Count')
    df_total = df_long.groupby('AppName').size().reset_index(name='Total')

# Merge count and total dataframes
df_percentage = pd.merge(df_count, df_total, on='AppName')
if category != 'App Usage Frequency':
    df_percentage['Percentage'] = (df_percentage['UserCount'] / df_percentage['TotalUsers']) * 100
else:
    df_percentage['Percentage'] = (df_percentage['Count'] / df_percentage['Total']) * 100

# Add total user base by category
df_percentage['TotalUserBase'] = df_percentage.groupby(category)['UserCount' if category != 'App Usage Frequency' else 'Count'].transform('sum')

# Add proportion of total users for each category
df_percentage['Proportion'] = (df_percentage['UserCount' if category != 'App Usage Frequency' else 'Count'] / df_percentage['TotalUserBase']) * 100

# Pivot the data to create a dataframe suitable for a bar chart
df_pivot = df_percentage.pivot(index='AppName', columns=category, values='Percentage').reindex(main_apps)

# Sort the DataFrame by the highest value in any column
df_pivot['max'] = df_pivot.max(axis=1)
df_pivot = df_pivot.sort_values(by='max', ascending=False).drop(columns='max').reset_index()

# Melt the pivot dataframe for plotting
df_melted = df_pivot.melt(id_vars='AppName', var_name=category, value_name='Percentage')

# Calculate additional insights for tooltips
df_insights = df_long.groupby(['AppName', category]).size().reset_index(name='Count')
df_insights = df_insights.merge(df_total.rename(columns={'TotalUsers': 'Total'}), on='AppName')
df_insights['Percentage'] = (df_insights['Count'] / df_insights['Total']) * 100

# Add total user base by category
df_insights['TotalUserBase'] = df_insights.groupby(category)['Count'].transform('sum')

# Add proportion of total users for each category
df_insights['Proportion'] = (df_insights['Count'] / df_insights['TotalUserBase']) * 100

# Merge insights with melted dataframe
df_final = pd.merge(df_melted, df_insights, on=['AppName', category], suffixes=('', '_insights'))

# Define color palettes for the categories
diverging_palette = {
    'Not Using': '#ff7f0e',  # Orange
    'Less often': '#fdae6b',  # Light orange
    'Several times a month': '#d9d9d9',  # Gray
    'Several times a week': '#a6cee3',  # Light blue
    'Daily': '#1f78b4',  # Blue
}

sequential_palette = {
    '18-24': '#e5f5e0',  # Light green
    '25-30': '#c7e9c0',  # Light-medium green
    '31-39': '#a1d99b',  # Medium green
    '40-49': '#74c476',  # Dark-medium green
    '50-64': '#41ab5d',  # Dark green
    '65+': '#238b45'  # Very dark green
}

income_palette = {
    'Low': '#edf8e9',  # Light green
    'Below Average': '#bae4b3',  # Light-medium green
    'Average': '#74c476',  # Medium green
    'Above Average': '#31a354',  # Dark-medium green
    'High': '#006d2c'  # Dark green
}

gender_palette = {
    'Male': '#1f77b4',  # Blue
    'Female': '#ff7f0e'  # Orange
}

religion_palette = {
    'Jewish - Traditional': '#e41a1c',  # Red
    'Jewish - Secular': '#377eb8',  # Blue
    'Jewish - National Religious': '#4daf4a',  # Green
    'Jewish - Orthodox': '#984ea3',  # Purple
    'Muslim': '#ff7f00'  # Orange
}

area_palette = {
    'Jerusalem': '#8dd3c7',  # Teal
    'Center': '#ffffb3',  # Yellow
    'Tel Aviv': '#bebada',  # Purple
    'North': '#fb8072',  # Coral
    'Haifa': '#80b1d3',  # Sky blue
    'Yehoda and Shomron': '#fdb462',  # Orange
    'South': '#b3de69',  # Light green
}

# Select appropriate color palette based on the category
color_discrete_map = diverging_palette if category == 'App Usage Frequency' else (
    sequential_palette if category == 'Age_group' else (
    income_palette if category == 'Income' else (
    gender_palette if category == 'Gender' else (
    religion_palette if category == 'Religion' else (
    area_palette if category == 'Area' else None)))))

# Plot the bar chart using Plotly
fig = px.bar(df_final,
             x='AppName', y='Percentage', color=category,
             hover_data={'AppName': False, 'Count': True, 'Total': True, 'Percentage': ':.2f',
                         'TotalUserBase': True, 'Proportion': ':.2f'},
             color_discrete_map=color_discrete_map)

# Add percentage labels inside bars and center them within each category color
fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside', insidetextanchor='middle')

fig.update_layout(
    title=f'App Usage by {category}',
    xaxis_title='App Name',
    yaxis_title='Percentage',
    yaxis=dict(showticklabels=True),  # Show y-axis labels
    legend_title_text=category,
    width=1400,  # Set the width of the plot
    height=800,  # Set the height of the plot
    xaxis={'categoryorder': 'total descending'}  # Sort by highest column
)

# Show the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Section 4: Heatmaps for App Usage by Religion
st.header('Heatmaps for App Usage by Religion')

# Select binary columns for heatmap
binary_columns = [col for col in df.columns if col.startswith('B-')]
binary_columns = [col for col in binary_columns if col.endswith((app1, app2))]

# Add validation to ensure there are enough binary columns
if len(binary_columns) < 2:
    st.error(f"Not enough binary columns found for the selected apps: {app1} and {app2}")
else:
    # Plot heatmaps for each selected app
    col1, col2 = st.columns(2)

    # Heatmap for the first app
    heatmap_data_app1 = pd.crosstab(df['Religion'], df[binary_columns[0]])
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data_app1, annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_title(f'Heatmap for {app1} Usage by Religion')

    # Heatmap for the second app
    heatmap_data_app2 = pd.crosstab(df['Religion'], df[binary_columns[1]])
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data_app2, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title(f'Heatmap for {app2} Usage by Religion')

    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
