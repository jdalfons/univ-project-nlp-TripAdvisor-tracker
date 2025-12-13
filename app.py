import streamlit as st
from streamlit_option_menu import option_menu
from utils.db import (
    get_downloaded_restaurants,
    init_db
)
from views.analytics import analytics_page
from views.home import home_page
from views.llm import llm_page
from views.restaurants import restaurant_page
import os
import pandas as pd
import base64

APP_TITLE = "TripAdvisor Scraper NLP"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="assets/img/Tripadvisor Icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

init_db()
df_downloaded_restaurants = get_downloaded_restaurants()

with st.sidebar:
    st.image("assets/img/Tripadvisor Icon_full.png")
    
    # KPIs in Sidebar
    st.metric("Restaurants Scrappés", len(df_downloaded_restaurants))

    # Sidebar Menu
    selected = option_menu(
        menu_title=None,
        options=["Accueil", "Restaurants", "Analytiques", "LLM"],
        icons=["house", "shop", "bar-chart", "robot"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#00AA6C"},
        }
    )

if selected == "Accueil":
    home_page()
elif selected == "LLM":
    llm_page(df_downloaded_restaurants)
elif selected == "Analytiques":
    analytics_page(df_downloaded_restaurants)
elif selected == "Restaurants":
    restaurant_page(df_downloaded_restaurants)

