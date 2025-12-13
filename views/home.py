""" This module contains the "Home" page. """

import streamlit as st


def home_page():
    """
    Renders the Home page.
    """
    st.title("TripAdvisor NLP Explorer 🚀")
    
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
            <p style="font-size: 18px; color: #31333F;">
                Bienvenue sur votre tableau de bord intelligent. Cet outil exploite la puissance du 
                <strong>Traitement du Langage Naturel (NLP)</strong> pour transformer les avis clients en insights stratégiques.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Fonctionnalités Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏪 Restaurants")
        st.info("Consultez des fiches détaillées, localisez les établissements sur la carte et téléchargez de nouveaux avis en temps réel.")
    
    with col2:
        st.markdown("### 📊 Analytics")
        st.success("Visualisez les tendances de sentiment, explorez les nuages de mots et identifiez les points forts/faibles via les KPI.")

    with col3:
        st.markdown("### 🤖 Assistant IA")
        st.warning("Interrogez notre LLM pour obtenir des résumés instantanés, des comparaisons et des réponses précises sur les avis.")

    st.divider()

    with st.expander("ℹ️ À propos du projet"):
        st.markdown(
            """
            **Moteur Technique :**
            - **Scraping** : Extraction massive via Beautiful Soup.
            - **NLP** : Analyse de sentiment (TextBlob), Émotions (NRCLex), Vectorisation (Word2Vec).
            - **IA** : Intégration de modèles génératifs (Mistral API).
            
            *Réalisé par Juan Diego Alfonso, Cyril Kocab et Maxence Liogier - Master 2 SISE.*
            """
        )
