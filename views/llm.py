"""
This module contains the "LLM" page with RAG Chat capabilities.
"""

import os
import streamlit as st
import time
from utils.db import get_reviews_one_restaurant
from utils.rag_chain import get_rag_chain, chat_with_guardrail

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

def llm_page(df):
    """
    Renders the LLM page as a RAG Chat.
    """
    # st.markdown("### 🤖 Chat avec les avis du Restaurant")

    # Layout for selector
    col1, col2 = st.columns([7, 2], vertical_alignment="bottom")
    
    with col1:
        restaurant_name = st.selectbox("Restaurant :", df["restaurant_name"].to_list())
    
    restaurant_id = df[df["restaurant_name"] == restaurant_name][
        "restaurant_id"
    ].values[0]

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_restaurant" not in st.session_state:
        st.session_state.current_restaurant = None

    # Reset chat if restaurant changes
    if st.session_state.current_restaurant != restaurant_name:
        st.session_state.messages = []
        st.session_state.current_restaurant = restaurant_name
        st.session_state.rag_chain = None
        # Add initial greeting
        st.session_state.messages.append({"role": "assistant", "content": f"Bonjour! Je m'appel Ollie et je suis prêt à répondre à vos questions sur **{restaurant_name}**."})

    # Display Chat History
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "assets/img/TripAdvisor Logo.webp" if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

    # Suggestions logic
    suggestion_input = None
    if len(st.session_state.messages) <= 1:
        st.write("💡 Suggestions :")
        col_s1, col_s2, col_s3 = st.columns(3)
        suggestions = ["Quelle est l'ambiance ?", "Quels sont les plats populaires ?", "Le service est-il bon ?"]
        if col_s1.button(suggestions[0], use_container_width=True):
             suggestion_input = suggestions[0]
        if col_s2.button(suggestions[1], use_container_width=True):
             suggestion_input = suggestions[1]
        if col_s3.button(suggestions[2], use_container_width=True):
             suggestion_input = suggestions[2]

    # User Input
    chat_input = st.chat_input("Posez une question sur ce restaurant...")
    
    prompt = chat_input or suggestion_input

    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant", avatar="assets/img/TripAdvisor Logo.webp"):
            with st.spinner("Analyse des avis en cours..."):
                try:
                    # Initialize Chain if needed
                    if st.session_state.get("rag_chain") is None:
                         filtered_df = get_reviews_one_restaurant(restaurant_id)
                         reviews = filtered_df["review_text"].tolist()
                         restaurant_info = df[df["restaurant_name"] == restaurant_name].to_dict(orient='records')[0]
                         st.session_state.rag_chain = get_rag_chain(reviews, str(restaurant_info))
                    
                    # Run Chain with Guardrail
                    api_key = os.getenv("MISTRAL_API_KEY")
                    response_dict = chat_with_guardrail(prompt, st.session_state.rag_chain, api_key)
                    response = response_dict["answer"]
                    
                    # Stream response
                    st.write_stream(stream_data(response))
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Une erreur est survenue: {e}")
                    if "MISTRAL_API_KEY" in str(e):
                         st.warning("Veuillez configurer MISTRAL_API_KEY dans les variables d'environnement.")
