""" This module contains the "Scraper" page. """

import time
import random
import re
import requests
import streamlit as st
from utils.tripAdvisorScraper import TripAdvisorSpecificRestaurantScraper
from utils.db import (
    save_reviews_to_db, 
    delete_reviews_by_restaurant_id,
    restaurant_exists,
    save_restaurant_to_db,
    get_reviews_info_by_restaurant)
from utils.functions import extract_types_from_df
import folium
from streamlit_folium import folium_static
                        
def verify_url(url):
    try:
        pattern = r"https?://www\.tripadvisor\.(com|fr)/Restaurant_Review-.+"
        match = re.match(pattern, url)
        if match:
            return re.sub(r"https?://www\.tripadvisor\.(com|fr)", "", url)
        
        raise ValueError("La URL n'est pas valide")
    except Exception as e:
        st.error(f"Erreur: {e}")
        return False
    
def scrape_restaurant_reviews(scraper, url, total_reviews_expected):
    """Scraper tous les avis pour un restaurant spécifique et afficher la progression dans Streamlit."""
    per.fetch_page(url)
    reviews = []
    page = 1
    tries = 0
    total_pages = total_reviews_expected // 15 + 5
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while scraper.url:
        time.sleep(random.uniform(1, 3))
        review_cards = scraper.get_review_cards()
        if not review_cards:
            tries += 1
            if tries > 10:
                raise Exception("Aucune carte de restaurant trouvée - Abandon")
            else:
                continue
        tries = 0
        for card in review_cards:
            reviews.append(scraper.parse_review(card))
        scraper.url = scraper.get_next_url()
        if scraper.url:
                scraper.fetch_page(scraper.url)
        page += 1
        progress_bar.progress(min(page / total_pages, 1.0))
        status_text.text(f"Scraping page {page} sur {total_pages}")
    
    progress_bar.progress(1.0)  # Assurez-vous que la barre de progression est complète à la fin
    status_text.text("Scraping terminé")
    return reviews

def download_restaurant_data(df):
    """
    Télécharger les données pour chaque restaurant dans le DataFrame filtré et enregistrer dans la base de données.
    """

    logs = []
    for i, (index, row) in enumerate(df.iterrows()):
        with st.spinner(f"Téléchargement des données pour {row['restaurant_name']}..."):
            time.sleep(random.uniform(1, 3))
            restaurant_url = row["restaurant_url"]
            restaurant_total_reviews = row["restaurant_total_reviews"]
            try:
                scraper = TripAdvisorSpecificRestaurantScraper()
                corpus = scrape_restaurant_reviews(scraper, restaurant_url, restaurant_total_reviews)
                save_reviews_to_db(row['restaurant_id'], corpus)
                logs.append(f"Succès: {row['restaurant_name']} - {len(corpus)} avis téléchargés.")
            except Exception as e:
                error_message = f"Erreur: {row['restaurant_name']} - {e}"
                logs.append(error_message)
                print(error_message)
                continue
        time.sleep(random.uniform(1, 3))
        logs.append({
            "restaurant_id": row["restaurant_id"],
            "restaurant_name": row["restaurant_name"],
            "reviews_downloaded": len(corpus)
        })
    
    with st.expander("Logs du processus"):
        for log in logs:
            st.write(log)


def restaurant_page(df):
    """
    Page Streamlit pour scraper les données des restaurants TripAdvisor.
    """

    st.title("Information des Restaurants")

    tab1, tab2 = st.tabs(["ℹ️ Info Restaurant", "⬇️ Télécharger Restaurant"])

    with tab1:
        restaurant_name = st.selectbox("Sélectionner un restaurant", df["restaurant_name"].unique())
        tab_info1, tab_info2 = st.tabs(["Info", "Mettre à jour"])
        
        with tab_info1:
            
            st.write(f"### Informations sur le restaurant {restaurant_name}")
            col1, col2 = st.columns(2)
            filtered_df = df[df["restaurant_name"] == restaurant_name]
            with col1:
                st.subheader("Restaurant Information")
                restaurant_info_dict = filtered_df.to_dict(orient='records')[0]
                st.write(f"**Nom du Restaurant:** {restaurant_info_dict['restaurant_name']}")
                st.write(f"**Prix du Restaurant:** {restaurant_info_dict['restaurant_price']}")
                st.markdown(f'<p><a href="https://www.tripadvisor.fr/{restaurant_info_dict["restaurant_url"]}" target="_blank">Visiter sur TripAdvisor</a></p>', unsafe_allow_html=True)
                st.write(f"**Type de Restaurant:** {restaurant_info_dict['restaurant_type']}")
                st.write(f"**Adresse:** {restaurant_info_dict['address']}")
            
            with col2:
                restaurant_id = filtered_df.iloc[0]["restaurant_id"]
                st.subheader("Reviews Information")
                reviews_info = get_reviews_info_by_restaurant(restaurant_id)
                st.write(f"**Reviews scraped:** {reviews_info['review_count'].iloc[0]}")
                st.write(f"**Average Rating:** {reviews_info['average_rating'].iloc[0]:.1f}")
                st.write(f"**First Comment Date:** {reviews_info['first_comment_date'].iloc[0]}")
                st.write(f"**Last Comment Date:** {reviews_info['last_comment_date'].iloc[0]}")
            st.subheader("À propos du Restaurant")  
            st.write(f"{restaurant_info_dict['restaurant_about']}")
            with tab_info2:
                st.warning("**Attention**: Les avis seront écrasés dans la base de données lors de la mise à jour.")
                try:
                    if not restaurant_name:
                        st.warning("Veuillez sélectionner un restaurant avant de continuer.")
                    else:
                        st.write(f"Mettre à jour les avis de {restaurant_name}")
                        if st.checkbox("Confirmer le risque de suppression des données existantes"):
                            if st.button("Télécharger", key="button_name_selection"):
                                st.write(f"Téléchargement des données pour {restaurant_name}...")
                                filtered_df = df[df["restaurant_name"] == restaurant_name]
                                restaurant_id = filtered_df.iloc[0]["restaurant_id"]
                                delete_reviews_by_restaurant_id(restaurant_id)
                                download_restaurant_data(filtered_df)
                            else:
                                st.warning("Veuillez confirmer la suppression des données existantes avant de continuer.")

                except FileNotFoundError:
                    st.write(
                        "Aucune donnée trouvée. Faites tourner le scraper avant de recommencer."
                    )   
                        
                
    with tab2:
        st.write("Scraper Page")
        form = st.form(key='scraper_form')
        
        url_input = form.text_input('URL', 'https://www.tripadvisor.fr/Restaurant_Review-g187265-d12419021-Reviews-L_Auberge_Des_Canuts-Lyon_Rhone_Auvergne_Rhone_Alpes.html')
        submit_button = form.form_submit_button('Submit')
        
        if submit_button:
            url = verify_url(url_input)
            if url:
                exist = restaurant_exists(url)
                if exist:
                    st.error("**Restaurant déjà existant**: Pour mettre à jour les informations, veuillez utiliser la page de mise à jour")
                else:
                    st.success("URL valide %s" % url_input)
                    if url != False:
                        while True:
                            try:
                                scraper = TripAdvisorSpecificRestaurantScraper()
                                scraper.fetch_page(url)
                                info = scraper.get_restaurant_info()
                                if info.get('restaurant_name') is not None:
                                    break
                            except Exception as e:
                                st.error(f"Erreur lors de la récupération des informations {e}")
                                time.sleep(5)
                        
                        with st.expander('Regarder les informations'):
                            st.write(info)
                        st.success("Restaurant enregistré sur la base de données")
                        
                        restaurant_df = save_restaurant_to_db(info)
                        download_restaurant_data(restaurant_df)
