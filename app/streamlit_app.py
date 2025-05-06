import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from transformers import pipeline
import warnings
import random

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Climate Change Impact Analysis", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df_raw = pd.read_csv('data/realistic_climate_change_impacts.csv')
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '')
    
    # Clean the 'economicimpact_usd' column
    if 'economicimpact_usd' in df.columns:
        df['economicimpact_usd'] = (
            df['economicimpact_usd']
            .str.replace(',', '', regex=True)
            .str.strip()
            .astype(float)
        )
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fill missing extreme weather events with 'Unknown'
    df['extremeweatherevent'] = df['extremeweatherevent'].fillna('Unknown')
    
    return df_raw, df

df_raw, df = load_data()

# Load Models
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
        st.warning(f"Model loading failed: {e}")
    return clustering_model, classification_model, scaler

clustering_model, classification_model, scaler = load_models()

# Initialize AI generator with error handling
try:
    generator = pipeline("text-generation", model="distilgpt2", framework="pt")  # Force PyTorch backend
except Exception as e:
    st.warning(f"AI analysis unavailable due to resource limits: {str(e)}")
    generator = None

# Sidebar Navigation
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
        "Word Cloud",
        "AI Analysis"
    ])

# Input Filters
col1, col2 = st.columns([3, 1])

with col2:
    st.title("Input Filters")
    co2_slider = st.slider(
        "CO₂ Level (ppm)",
        float(df['co2level_ppm'].min()),
        float(df['co2level_ppm'].max()),
        float(df['co2level_ppm'].mean()),
        key="co2_slider"
    )
    temp_slider = st.slider(
        "Temperature Anomaly (°C)",
        float(df['temperatureanomaly_c'].min()),
        float(df['temperatureanomaly_c'].max()),
        float(df['temperatureanomaly_c'].mean()),
        key="temp_slider"
    )
    econ_slider = st.slider(
        "Economic Impact (USD)",
        float(df['economicimpact_usd'].min()),
        float(df['economicimpact_usd'].max()),
        float(df['economicimpact_usd'].mean()),
        key="econ_slider"
    )
    pop_slider = st.slider(
        "Population Affected",
        int(df['populationaffected'].min()),
        int(df['populationaffected'].max()),
        int(df['populationaffected'].mean()),
        key="pop_slider"
    )

# Apply Filters
filtered_df = df[
    (df['co2level_ppm'] >= co2_slider) &
    (df['temperatureanomaly_c'] >= temp_slider) &
    (df['economicimpact_usd'] >= econ_slider) &
    (df['populationaffected'] >= pop_slider)
].copy()

# Generate fake event descriptions for WordCloud (simulating notebook)
words = ["hurricane", "flood", "wildfire", "heatwave", "drought", "storm", "rainfall", "temperature", "climate", "disaster", "emergency"]
filtered_df['event_description'] = filtered_df['extremeweatherevent'].apply(
    lambda event: " ".join(random.choices(words, k=random.randint(5, 10))) + f" {event.lower()}"
)

# Content
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
        if filtered_df.empty or clustering_model is None or scaler is None:
            st.warning("No data or models available for clustering. Please adjust filters or load models.")
        else:
            features = scaler.transform(filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']])
            cluster_labels = clustering_model.predict(features)
            filtered_df['Cluster'] = cluster_labels
            fig = px.scatter(filtered_df, x='co2level_ppm', y='temperatureanomaly_c', color='Cluster',
                             hover_data=['country', 'extremeweatherevent'])
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Classification":
        st.subheader("Event Classification Prediction")
        if filtered_df.empty or classification_model is None or scaler is None:
            st.warning("No data or models available for classification. Please adjust filters or load models.")
        else:
            features = scaler.transform(filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']])
            predictions = classification_model.predict(features)
            filtered_df['Predicted Event'] = predictions
            fig = px.histogram(filtered_df, x='Predicted Event')
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Feature Importance":
        st.subheader("Feature Importance")
        if classification_model:
            features = scaler.transform(filtered_df[['co2level_ppm', 'temperatureanomaly_c', 'economicimpact_usd', 'populationaffected']])
            importance = classification_model.feature_importances_
            features_names = ['CO₂ Level (ppm)', 'Temperature Anomaly (°C)', 'Economic Impact (USD)', 'Population Affected']
            fig = px.bar(x=features_names, y=importance, labels={'x': 'Feature', 'y': 'Importance'})
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
        text = " ".join(filtered_df['event_description'].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma', contour_width=1, contour_color='steelblue').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    elif page == "AI Analysis":
        st.subheader("AI-Generated Climate Impact Analysis")
        selected_country = st.selectbox("Select Country:", filtered_df['country'].unique(), key="ai_country")
        calc_method = st.selectbox("Calc Method:", ["per year", "per day", "square foot"], key="ai_method")
        if st.button("Generate AI Analysis"):
            if selected_country not in df['country'].values:
                avg_co2 = 0
            else:
                avg_co2 = df[df['country'] == selected_country]['co2level_ppm'].mean()
            prompt = (
                f"The CO2 consumption of {selected_country} is {avg_co2:.2f} ppm. This means "
                f"the country emits substantial greenhouse gases per capita. "
                f"Would you like to calculate the economic impact of that consumption "
                f"per square foot, per day, or per year?"
            )
            if generator:
                try:
                    result = generator(prompt, max_length=100, num_return_sequences=1)
                    st.markdown(f"### AI-Generated Summary for {selected_country}")
                    st.markdown(result[0]['generated_text'])
                except Exception as e:
                    st.error(f"Failed to generate AI summary: {str(e)}")
            else:
                st.write("AI analysis is currently unavailable due to resource constraints.")

            def calculate_impact(df, country, method):
                row = df[df['country'] == country]
                if row.empty:
                    return "No data available for this country."
                econ = row['economicimpact_usd'].values[0]
                if pd.isna(econ):
                    return "No valid economic impact data available."
                if method.lower() == "square foot":
                    impact = econ / 1000000
                    return f"Estimated economic impact per square foot: ${impact:,.2f}"
                elif method.lower() == "per day":
                    impact = econ / 365
                    return f"Estimated daily economic impact: ${impact:,.2f}"
                elif method.lower() == "per year":
                    return f"Annual economic impact: ${econ:,.2f}"
                return "Please enter one of: square foot, per day, or per year."

            impact = calculate_impact(df, selected_country, calc_method)
            prompt = f"The CO2 consumption of {selected_country} is calculated using the {calc_method} method. Provide an analysis."
            if generator:
                try:
                    result = generator(prompt, max_length=100, num_return_sequences=1)
                    st.markdown(f"### Detailed Analysis")
                    st.markdown(result[0]['generated_text'])
                except Exception as e:
                    st.error(f"Failed to generate detailed analysis: {str(e)}")
            else:
                st.write("Detailed analysis is currently unavailable due to resource constraints.")
            st.markdown(f"**{impact}**")
