""" Ce module contient la page "Analyse". """

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

from utils.db import get_downloaded_restaurants, get_restaurant_by_id
from collections import Counter
import altair as alt

from utils.functions import (
    extract_types_from_df,
    generate_wordcloud,
    clean_text_df,
    generate_word2vec,
    generate_sentiments_analysis,
    generate_word_frequencies_chart,
    generate_spider_plot,
    generate_evolution_chart,
    generate_rating_distribution_chart,
    generate_ngrams_chart,
    generate_pos_distribution_chart
)


# Fonction de filtrage des restaurants
def restaurant_filters(df, tab_title):
    """
    Fonction pour filtrer les restaurants par type, prix et nom.
    """
    col1, col2 = st.columns(2)

    with col1:
        # Filtrer par type
        types = extract_types_from_df(df, True)
        # st.write(types)
        types = ["Tous"] + list(types)
        selected_type = st.selectbox(
            "Sélectionnez le type de restaurant",
            types,
            key=f"restaurant_type_{tab_title}",
        )

    with col2:
        # Filtrer par prix
        prices = ["Tous"] + df["restaurant_price"].unique().tolist()
        selected_price = st.selectbox(
            "Sélectionnez le prix du restaurant",
            prices,
            key=f"restaurant_price_{tab_title}",
        )

    # Filtrer par nom
    if selected_type != "Tous" and selected_price != "Tous":
        names = ["Tous"] + df[
            (df["restaurant_type"].str.contains(selected_type, case=False, na=False))
            & (df["restaurant_price"] == selected_price)
        ]["restaurant_name"].unique().tolist()
    elif selected_type != "Tous":
        names = ["Tous"] + df[
            df["restaurant_type"].str.contains(selected_type, case=False, na=False)
        ]["restaurant_name"].unique().tolist()
    elif selected_price != "Tous":
        names = ["Tous"] + df[df["restaurant_price"] == selected_price][
            "restaurant_name"
        ].unique().tolist()
    else:
        names = ["Tous"] + df["restaurant_name"].unique().tolist()

    selected_names = st.multiselect(
        "Sélectionnez les noms des restaurants",
        names,
        key=f"restaurant_names_{tab_title}",
    )
    return selected_names, names


def get_filtered_restaurant(df, selected_names, names, relevance):
    """
    Fonction pour filtrer les restaurants sélectionnés par l'utilisateur.
    """
    if len(selected_names) <= 0 and "Tous" not in selected_names:
        st.warning("Veuillez sélectionner au moins un restaurant.")
        st.stop()
    else:
        filtered_df = df.copy()
        if "Tous" not in selected_names:
            filtered_df = filtered_df[
                filtered_df["restaurant_name"].isin(selected_names)
            ]
        else:  # 'Tous' in selected_names
            names = [
                item for item in names if "Tous" not in item
            ]  # On supprime "Tous" des noms restants
            filtered_df = filtered_df[filtered_df["restaurant_name"].isin(names)]
        if len(filtered_df) > 10:
            st.warning(
                "Vous avez sélectionné plus de dix restaurants, cela peut prendre du temps."
            )

        # Obtenir les détails des restaurants par IDs
        restaurant_ids = filtered_df["restaurant_id"].tolist()
        filtered_df = get_restaurant_by_id(restaurant_ids)
        if relevance:
            avg = filtered_df["contributions"].median()
            filtered_df = filtered_df[filtered_df["contributions"] >= avg]
        return filtered_df


def analytics_page(df):
    """Page d'analyse des restaurants (Dashboard)."""

    st.title("Tableau de Bord Analytique 📊")
    st.markdown("Vue d'ensemble des performances et des sentiments des restaurants.")

    # Keeping filters at the top level
    with st.expander("Filtres et Sélection", expanded=True):
        TAB_TITLE = "Dashboard"
        selected_names, names = restaurant_filters(df, TAB_TITLE)
        relevance = st.checkbox("Avis pertinents uniquement (Top contributeurs)", value=False,
                                help="Seuls les avis des contributeurs au-dessus de la médiane sont inclus.",
                                key=f"relevant_only_{TAB_TITLE}")

    if st.button("Actualiser le Tableau de Bord", type="primary", key="refresh_dashboard"):
        with st.spinner("Analyse en cours... ⏳"):
            filtered_df = get_filtered_restaurant(df, selected_names, names, relevance)
            filtered_df = clean_text_df(filtered_df)
            
            # --- CALCUL DES KPIs ---
            total_reviews = len(filtered_df)
            avg_rating = filtered_df["rating"].mean()
            
            # Generate sentiment analysis first to get 'sentiment' column
            emotions_par_resto, scatter_plot = generate_sentiments_analysis(filtered_df)
            avg_sentiment = filtered_df["sentiment"].mean()
            
            num_restaurants = filtered_df["restaurant_id"].nunique()

            # --- AFFICHAGE DES KPIs ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Nombre d'Avis", total_reviews)
            kpi2.metric("Note Moyenne", f"{avg_rating:.2f}/5")
            kpi3.metric("Score Sentiment", f"{avg_sentiment:.2f}", help="De -1 (Négatif) à +1 (Positif)")
            kpi4.metric("Restaurants", num_restaurants)

            st.divider()

            # --- GRAPHIQUES LIGNE 1 : Evolution & Distribution ---
            st.subheader("Tendances et Distribution")
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                st.markdown("#### 📅 Évolution Temporelle")
                evolution_chart = generate_evolution_chart(filtered_df)
                st.altair_chart(evolution_chart, use_container_width=True)
                
            with row1_col2:
                st.markdown("#### ⭐ Distribution des Notes")
                dist_chart = generate_rating_distribution_chart(filtered_df)
                st.altair_chart(dist_chart, use_container_width=True)

            st.divider()

            # --- GRAPHIQUES LIGNE 2 : Sentiment vs Note & Emotions ---
            st.subheader("Analyse Approfondie")
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                st.markdown("#### 🙂 Sentiment vs Note")
                st.plotly_chart(scatter_plot, use_container_width=True)
                
            with row2_col2:
                st.markdown("#### 🎭 Radar des Émotions")
                spider_plot = generate_spider_plot(emotions_par_resto)
                st.plotly_chart(spider_plot, use_container_width=True)

            # --- ANALYSES TEXTUELLES (NLP) ---
            st.divider()
            st.subheader("Analyses Textuelles Avancées (NLP)")
            
            row3_col1, row3_col2 = st.columns(2)
            
            with row3_col1:
                st.markdown("#### 🗣️ Expressions Fréquentes (Bigrammes)")
                ngram_chart = generate_ngrams_chart(filtered_df, n=2)
                st.altair_chart(ngram_chart, use_container_width=True)
                
            with row3_col2:
                st.markdown("#### 📝 Grammaire (Adjectifs vs Noms)")
                pos_chart = generate_pos_distribution_chart(filtered_df)
                st.altair_chart(pos_chart, use_container_width=True)
                
            st.markdown("#### ☁️ Nuage de Mots")
            # Wordcloud is matplotlib figure
            ignored_words = [] # Could add filter input if needed
            wc = generate_wordcloud(filtered_df, ignored_words)
            fig_wc = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig_wc)

            st.divider()

            # --- DERNIERS AVIS ---
            st.subheader("Derniers Avis Analysés")
            # Sort by date if available, otherwise just take head
            if 'date' in filtered_df.columns:
                recent_reviews = filtered_df.sort_values(by='date', ascending=False).head(5)
            else:
                recent_reviews = filtered_df.head(5)
            
            for index, row in recent_reviews.iterrows():
                with st.container():
                     cols = st.columns([1, 4])
                     with cols[0]:
                         st.write(f"**{row['rating']} ⭐**")
                         sentiment_score = row.get('sentiment', 0)
                         if sentiment_score > 0.05:
                             st.caption("🟢 Positif")
                         elif sentiment_score < -0.05:
                             st.caption("🔴 Négatif")
                         else:
                             st.caption("⚪ Neutre")
                     with cols[1]:
                         st.write(f"_{row['restaurant_name']}_")
                         st.write(f"\"{row['review_text'][:300]}...\"")
                         st.caption(f"Date: {row.get('date', 'N/A')}")
                st.markdown("---")

