import streamlit as st
from streamlit_option_menu import option_menu
from utils.db import (
    get_downloaded_restaurants)
from views.analytics import analytics_page
from views.home import home_page
from views.llm import llm_page
from views.restaurants import restaurant_page
from views.map import map_page
# import psycopg2
# import os
# import pandas as pd

APP_TITLE = "TripAdvisor Scraper NLP"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="assets/img/Tripadvisor Icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image("assets/img/Tripadvisor Icon_full.png")

    selected = option_menu(
        menu_title='',
        options=["Accueil", "Restaurants", "Analytiques", "LLM", "Carte"],
        icons=["house", "houses", "bar-chart", "robot", "map"],
        default_index=0,
        # orientation="horizontal",
    )

df_downloaded_restaurants = get_downloaded_restaurants()

if selected == "Accueil":
    home_page()
elif selected == "LLM":
    llm_page(df_downloaded_restaurants)
elif selected == "Analytiques":
    analytics_page(df_downloaded_restaurants)
elif selected == "Restaurants":
    restaurant_page(df_downloaded_restaurants)
elif selected == "Carte":
    map_page(df_downloaded_restaurants)

