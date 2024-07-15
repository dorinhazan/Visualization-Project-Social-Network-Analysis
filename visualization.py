import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="Analyzing Social Platforms Usage Patterns",
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
st.title('Analyzing Social Network and Online Service Usage Patterns')

# Select category for analysis
category = st.selectbox('Select a Category from the following options:', ['Age_group', 'Gender', 'Income', 'App Usage Frequency'])

# Create a copy of the original dataframe to avoid altering it
df_original = df.copy()

# Extract relevant columns for the selected category
if category == 'Age_group':
    category_order = ['18-24', '25-30', '31-39', '40-49', '50-64', '65+']
    df[category] = pd.Categorical(df[category], categories=category_order, ordered=True)
elif category == 'App Usage Frequency':
    frequency_columns = [col for col in df_original.columns if col.startswith('Frequency-')]
    df_long_freq = pd.melt(df_original, id_vars=['Gender', 'Income'], value_vars=frequency_columns,
                           var_name='AppName', value_name='Frequency')
    df_long_freq['AppName'] = df_long_freq['AppName'].str.replace('Frequency-', '')
    main_apps = ['Facebook', 'YouTube', 'Instagram', 'TikTok', 'Twitter', 'LinkedIn', 'WhatsApp']
    df_long_freq = df_long_freq[df_long_freq['AppName'].isin(main_apps)]
    df_long_freq = df_long_freq.rename(columns={'Frequency': category})
    df = df_long_freq
    category_order = ['Not Using', 'Less often', 'Several times a month', 'Several times a week', 'Daily']
    df[category] = pd.Categorical(df[category], categories=category_order, ordered=True)
elif category == 'Income':
    category_order = ['Way below Average', 'below Average', 'Similar to Average', 'Above Average', 'Way Above Average']
    df[category] = pd.Categorical(df[category], categories=category_order, ordered=True)
else:  # Gender
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

diverging_palette = {
    'Not Using': '#deebf7',  # Light blue
    'Less often': '#9ecae1',  # Light-medium blue
    'Several times a month': '#3182bd',  # Medium blue
    'Several times a week': '#08519c',  # Dark-medium blue
    'Daily': '#08306b',  # Dark blue
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
    'Way below Average': '#fd8d3c',  # Light orange
    'below Average': '#fdae6b',  # Light-medium orange
    'Similar to Average': '#ffeda0',  # Light yellow
    'Above Average': '#3182bd',  # Medium blue
    'Way Above Average': '#08306b'  # Dark blue
}

gender_palette = {
    'Male': '#1f77b4',  # Blue
    'Female': '#ff7f0e'  # Pink
}

# Select appropriate color palette based on the category
color_discrete_map = diverging_palette if category == 'App Usage Frequency' else (
    sequential_palette if category == 'Age_group' else (
    income_palette if category == 'Income' else (
    gender_palette if category == 'Gender' else None)))

# Plot the bar chart using Plotly
fig = px.bar(df_final,
             x='AppName', y='Percentage', color=category,
             hover_data={'AppName': False, 'Count': True,
                         'Total': True, 'Percentage': ':.2f',
                         'TotalUserBase': True, 'Proportion': ':.2f'},
             color_discrete_map=color_discrete_map)

# Add percentage labels inside bars and center them within each category color
fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside', insidetextanchor='middle')

fig.update_layout(
    title=f'Application usage categorized by {category}',
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


# Streamlit app
st.title('Visualizing Application Usage Patterns Across Two Applications')

# Section 1: App Selection
st.header('Select your two main Apps for Analysis')

# Dropdowns for selecting two apps
app_options = ['Facebook', 'Instagram', 'YouTube', 'Twitter']
# Place dropdowns for selecting apps side by side
col1, col2 = st.columns(2)

with col1:
    app1 = st.selectbox('Select first app', app_options)

with col2:
    # Filter out the first app from the options for the second dropdown
    app_options_filtered = [app for app in app_options if app != app1]
    app2 = st.selectbox('Select second app', app_options_filtered)

# Filter dataframe for selected apps
selected_apps = [f'Frequency-{app1}-num', f'Frequency-{app2}-num']

# Section 2: Scatter Plots
st.header(f'Scatter Plots Illustrating the Relationship Between Age, Income, and Usage Frequency')

# Custom color scale from light to dark
custom_color_scale = [
    [0, "yellow"],
    [1, "darkred"]
]

# Scatter plot for the first app
fig_app1 = px.scatter(
    df,
    x='Age_group-num',
    y='Income-num',
    color=selected_apps[0],
    title=f'{app1}',
    labels={selected_apps[0]: f'{app1} Frequency'},
    color_continuous_scale=custom_color_scale
)

fig_app1.update_layout(
    xaxis_title='Age group',
    yaxis_title='Income'
)

# Scatter plot for the second app
fig_app2 = px.scatter(
    df,
    x='Age_group-num',
    y='Income-num',
    color=selected_apps[1],
    title=f'{app2}',
    labels={selected_apps[1]: f'{app2} Frequency'},
    color_continuous_scale=custom_color_scale
)

fig_app2.update_layout(
    xaxis_title='Age group',
    yaxis_title='Income'
)


col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(fig_app1, use_container_width=True)

with col2:
    st.plotly_chart(fig_app2, use_container_width=True)


# Filter dataframe for selected apps
selected_apps = [f'Frequency-{app1}-num', f'Frequency-{app2}-num']

# Section 2: Scatter Plots
st.header(f'Scatter Plots Illustrating the Relationship Between Age, Income, and Usage Frequency')

# Custom color scale from light to dark
custom_color_scale = [
    [0, "yellow"],
    [1, "darkred"]
]

# Scatter plot for the first app
fig_app1 = px.scatter(
    df,
    x='Age_group-num',
    y='Income-num',
    color=selected_apps[0],
    title=f'{app1}',
    labels={selected_apps[0]: f'{app1} Frequency'},
    color_continuous_scale=custom_color_scale
)

fig_app1.update_layout(
    xaxis_title='Age group',
    yaxis_title='Income'

)

# Scatter plot for the second app
fig_app2 = px.scatter(
    df,
    x='Age_group-num',
    y='Income-num',
    color=selected_apps[1],
    title=f'{app2}',
    labels={selected_apps[1]: f'{app2} Frequency'},
    color_continuous_scale=custom_color_scale
)

fig_app2.update_layout(
    xaxis_title='Age group',
    yaxis_title='Income'

)


col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(fig_app1, use_container_width=True)

with col2:
    st.plotly_chart(fig_app2, use_container_width=True)





# Section 4: Heatmaps for App Usage by Religion
st.header(f'Heatmaps Showing the Correlation Between Religion and Application Usage')

# Select binary columns for heatmap
binary_columns = [col for col in df.columns if col.startswith('B-')]
binary_columns = [col for col in binary_columns if col.endswith((app1, app2))]

# Add validation to ensure there are enough binary columns
if len(binary_columns) < 2:
    st.error(f"Not enough binary columns found for the selected apps: {app1} and {app2}")
else:
    # Clean column names for heatmap
    app1_cleaned = app1.replace('B-', '')
    app2_cleaned = app2.replace('B-', '')

    df.rename(columns={binary_columns[0]: app1_cleaned, binary_columns[1]: app2_cleaned}, inplace=True)

    # Heatmap for the first app
    heatmap_data_app1 = pd.crosstab(df['Religion'],df[app1_cleaned])
    heatmap_data_app1.columns = ['Not Using', 'Using']
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data_app1, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'{app1_cleaned}')

    # Heatmap for the second app
    heatmap_data_app2 = pd.crosstab(df['Religion'],df[app2_cleaned])
    heatmap_data_app2.columns = ['Not Using', 'Using']
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data_app2, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'{app2_cleaned}')

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
# Section 5: Pie Charts for Family Status of Users
st.header(f'Pie Charts Showing Family Status of Users in {app1} and {app2}')

if 'FamilyStatus' in df.columns and 'Facebook' in df.columns and 'Instagram' in df.columns:
    # Filter dataframe for the selected apps
    family_status_columns = ['FamilyStatus', 'Facebook', 'Instagram']
    df_family_status = df[family_status_columns]

    # Filter out "Other" category
    df_family_status = df_family_status[df_family_status['FamilyStatus'] != 'Other']

    # Define custom colors
    custom_colors = ["#FFCD00", "#008FC4", "#1AB394", "#FFA000", "#FF3D00"]

    # Function to create sorted pie chart using Matplotlib
    def create_sorted_pie_chart(data, title):
        # Sort data from highest to lowest
        data = data.sort_values(ascending=False)
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(data, autopct=lambda p: f'{p:.1f}%', startangle=90, counterclock=False, colors=custom_colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title(title)
        ax.legend(wedges, data.index, title="Family Status", loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
        return fig

    # Pie chart for Facebook
    family_status_facebook = df_family_status[df_family_status['Facebook'] == 1]['FamilyStatus'].value_counts()
    fig_pie1 = create_sorted_pie_chart(family_status_facebook, 'Facebook')

    # Pie chart for Instagram
    family_status_instagram = df_family_status[df_family_status['Instagram'] == 1]['FamilyStatus'].value_counts()
    fig_pie2 = create_sorted_pie_chart(family_status_instagram, 'Instagram')

    # Display the pie charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_pie1)
    with col2:
        st.pyplot(fig_pie2)
