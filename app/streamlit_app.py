# ===================== Libraries =====================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
import os

# ===================== Page Configuration =====================
st.set_page_config(page_title="Climate Change Impact Analysis", layout="wide")

# ===================== Load Data =====================
@st.cache_data
def load_data():
    df_raw = pd.read_csv('data/realistic_climate_change_impacts.csv')
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '')
    
    # Clean the 'economicimpact_usd' column
    if 'economicimpact_usd' in df.columns:
        df['economicimpact_usd'] = (
            df['economicimpact_usd']
            .str.replace(',', '', regex=True)  # Remove commas
            .str.strip()                      # Remove leading/trailing spaces
            .astype(float)                    # Convert to float
        )
    
    return df_raw, df

df_raw, df = load_data()

# ===================== Load Models =====================
@st.cache_resource
def load_models():
    clustering_model = None
    classification_model = None
    scaler = None
    try:
        clustering_model = joblib.load('models/clustering_model.pkl')
        classification_model = joblib.load('models/classification_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except Exception as e:
        pass  # models optional
    return clustering_model, classification_model, scaler

clustering_model, classification_model, scaler = load_models()

# ===================== Sidebar Navigation =====================
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", [
        "Raw Data",
        "Cleaned Data",
        "Top Countries",
        "Heatmap",
        "Clustering",
        "Classification",
        "Feature Importance",
        "World Map",
        "Word Cloud"
    ])

# ===================== Input Filters =====================
# Create two columns: one for content and one for filters
col1, col2 = st.columns([3, 1])

# Column 2: Filters
with col2:
    st.title("Input Filters")
    co2_slider = st.slider(
        "CO₂ Level (ppm)",
        float(df['co2level_ppm'].min()),
        float(df['co2level_ppm'].max()),
        float(df['co2level_ppm'].mean()),
        key="co2_slider"  # Unique key
    )
    temp_slider = st.slider(
        "Temperature Anomaly (°C)",
        float(df['temperatureanomaly_c'].min()),
        float(df['temperatureanomaly_c'].max()),
        float(df['temperatureanomaly_c'].mean()),
        key="temp_slider"  # Unique key
    )
    econ_slider = st.slider(
        "Economic Impact (USD)",
        float(df['economicimpact_usd'].min()),
        float(df['economicimpact_usd'].max()),
        float(df['economicimpact_usd'].mean()),
        key="econ_slider"  # Unique key
    )
    pop_slider = st.slider(
        "Population Affected",
        int(df['populationaffected'].min()),
        int(df['populationaffected'].max()),
        int(df['populationaffected'].mean()),
        key="pop_slider"  # Unique key
    )

# Apply Filters
filtered_df = df[
    (df['co2level_ppm'] >= co2_slider) &
    (df['temperatureanomaly_c'] >= temp_slider) &
    (df['economicimpact_usd'] >= econ_slider) &
    (df['populationaffected'] >= pop_slider)
]

# Column 1: Content
with col1:
    st.title(page)

    if page == "Raw Data":
        st.subheader("Raw Dataset Overview (Before Cleaning)")
        st.dataframe(df_raw, use_container_width=True)

    elif page == "Cleaned Data":
        st.subheader("Cleaned Dataset Overview (After Cleaning)")
        st.dataframe(filtered_df, use_container_width=True)

    elif page == "Top Countries":
        st.subheader("Top 10 Countries by Number of Events")
        top_countries = filtered_df['country'].value_counts().nlargest(10)
        fig = px.bar(top_countries, x=top_countries.index, y=top_countries.values,
                     labels={'x': 'Country', 'y': 'Number of Events'})
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Heatmap":
        st.subheader("Correlation Heatmap")
        corr = filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Clustering":
        st.subheader("Clustering Analysis")
        if filtered_df.empty:
            st.warning("No data available for clustering. Please adjust your filters.")
        else:
            try:
                features = scaler.transform(filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']])
                cluster_labels = clustering_model.predict(features)
                filtered_df['Cluster'] = cluster_labels
                fig = px.scatter(filtered_df, x='co2level_ppm', y='temperatureanomaly_c', color='Cluster',
                                 hover_data=['country', 'extremeweatherevent'])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif page == "Classification":
        st.subheader("Event Classification Prediction")
        if filtered_df.empty:
            st.warning("No data available for classification. Please adjust your filters.")
        else:
            try:
                features = scaler.transform(filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']])
                predictions = classification_model.predict(features)
                filtered_df['Predicted Event'] = predictions
                fig = px.histogram(filtered_df, x='Predicted Event')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    elif page == "Feature Importance":
        st.subheader("Feature Importance")
        if classification_model:
            # Use filtered_df to calculate feature importance
            features = scaler.transform(filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']])
            importance = classification_model.feature_importances_
            features_names = ['CO₂ Level (ppm)', 'Temperature Anomaly (°C)', 'Economic Impact (USD)', 'Population Affected']
            
            # Create the bar chart using Plotly Express
            fig = px.bar(x=features_names, y=importance,
                         labels={'x': 'Feature', 'y': 'Importance'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available.")

    elif page == "World Map":
        st.subheader("World Map of Event Counts")
        try:
            import geopandas as gpd
            world = gpd.read_file("data/ne_110m_admin_0_countries.shp")
            merged = world.merge(filtered_df.groupby('country').size().reset_index(name='counts'),
                                  how='left', left_on='NAME', right_on='country')
            fig = px.choropleth(merged, geojson=merged.geometry, locations=merged.index, color='counts', projection="natural earth")
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"World map loading error: {e}")

    elif page == "Word Cloud":
        st.subheader("Word Cloud of Events")
        text = " ".join(filtered_df['extremeweatherevent'].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)