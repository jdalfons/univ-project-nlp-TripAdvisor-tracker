import re
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter
import altair as alt
import requests
from nrclex import NRCLex
from nltk.util import ngrams
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger_eng')
from nltk.util import ngrams
from nltk import pos_tag  # Add this import

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger_eng') # For POS tagging (English by default, but we might need French tagger or mapping)
# Note: NLTK's default pos_tag is for English. For French, we usually need 'averaged_perceptron_tagger' 
# or a specific French model. However, 'pos_tag' uses Penn Treebank tagset which is English-centric.
# For simplicity in this demo, we might rely on simple split or if user has french setup. 
# Let's assume standard pos_tag for now or check if we need spacy. 
# Given the environment, let's try standard NLTK pos_tag (it might be approximate for French but shows the concept).
from nltk.corpus import stopwords


def get_keys():
    mistral_key = os.getenv('MISTRAL_KEY')
    google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    return {
        'mistral_key': mistral_key,
        'google_maps_api_key': google_maps_api_key
    }
    
    
words_not_relevant = [
    "a", "à", "au", "aux", "avec", "ce", "ces", "dans",
    "très", "trop", "peu", "beaucoup", "restaurant", "plu",
    "peu", "beaucoup", "plus", "c'est", "cuisine",
    "très", "sans", "plus", "bien", "trop", "si", "cette"
    "a", "à", "au", "aux", "avec", "ce", "ces", "dans", 
    "de", "des"
]
    
def clean_text(text: str) -> str:
    """
    Clean the input text by removing newlines, carriage returns, and tabs.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    txt = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    txt = txt.replace("  ", " ")
    return txt.strip()


def extract_types_from_df(df, original_columns=False):
    """
    Extract unique restaurant types from the DataFrame.

    Args:
        df (pd.DataFrame): The input dataframe containing 'restaurant_type' column.
        original_columns (bool): If True, return a dictionary with types and all that match.

    Returns:
        list or dict: A list of unique restaurant types or a dictionary with types and all that match.
    """
    rest_types = list()
    try:
        df["restaurant_type"] = df["restaurant_type"].apply(
            lambda x: None if "€" in str(x) else x
        )
        temp_rest_types = df["restaurant_type"].dropna().unique()
        for rest_type in temp_rest_types:
            types = rest_type.split(",")
            for type in types:
                rest_types.append(type.strip())
        rest_types = list(set(rest_types))
        rest_types.sort()

        if original_columns:
            type_dict = {
                type: df[df["restaurant_type"].str.contains(type, na=False)][
                    "restaurant_type"
                ].tolist()
                for type in rest_types
            }
            return type_dict

        return rest_types
    except KeyError:
        return rest_types
    except Exception as e:
        print(f"Error: {e}")
        return rest_types


def extract_by_regex(text: str, regex: str) -> str:
    """
    Extract a substring from the input text using a regular expression.

    Args:
        text (str): The input text.
        regex (str): The regular expression pattern.

    Returns:
        str: The extracted substring or an empty string if no match is found.
    """
    pattern = re.compile(regex)
    match = pattern.search(text)
    if match:
        if match.groups():
            return (
                match.group(1) + " " + match.group(2)
                if len(match.groups()) > 1
                else match.group(1)
            )
        else:
            return match.group(0)
    return ""


def filter_by_regex(text: str, pattern: str) -> str:
    """
    Filter the input text by removing substrings that match the regular expression pattern.

    Args:
        text (str): The input text.
        pattern (str): The regular expression pattern.

    Returns:
        str: The filtered text or None if no match is found.
    """
    match = re.sub(pattern, "", text)
    return match if match else None


def clean_text_df(df: pd.DataFrame, root_type: str = "lemmatization") -> pd.DataFrame:
    """
    Clean the text in the dataframe by removing stop words and applying stemming or lemmatization.

    Args:
        df (pd.DataFrame): The input dataframe containing a 'review_text' column.
        root_type (str): The type of root processing to apply ('stemming' or 'lemmatization').

    Returns:
        pd.DataFrame: The dataframe with an additional 'cleaned_text' column.
    """
    stop_words = set(stopwords.words("french"))
    stop_words.update(words_not_relevant)
    stemmer = SnowballStemmer("french")
    lemmatizer = nltk.WordNetLemmatizer()

    def process_text(text):
        tokens = nltk.word_tokenize(text, language="french")
        if root_type == "stemming":
            tokens = [
                stemmer.stem(word.lower()).lower() for word in tokens if word.lower() not in stop_words
            ]
        else:
            tokens = [
                lemmatizer.lemmatize(word.lower()).lower()
                for word in tokens
                if word.lower() not in stop_words
            ]
        return " ".join(tokens)

    # Drop rows where 'review_text' is NaN or None
    df = df.dropna(subset=['review_text'])
    
    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')

    df["cleaned_text"] = df["review_text"].apply(lambda x: process_text(x))

    return df


def generate_wordcloud(df: pd.DataFrame, ignored_words=list()) -> WordCloud:
    """
    Generate a word cloud from the cleaned text in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing 'restaurant_name' and 'cleaned_text' columns.

    Returns:
        WordCloud: The generated word cloud.
    """
    if not {"restaurant_name", "cleaned_text"}.issubset(df.columns):
        raise ValueError(
            "Dataframe must contain 'restaurant_name' and 'cleaned_text' columns."
        )
    
    # Join the filtered tokens back into a string
    text = " ".join(review for review in df["cleaned_text"])
    text_separed = text.lower().split(" ")
    print(len(text_separed))
    text_separed = [word for word in text_separed if word not in words_not_relevant]
    print(len(text_separed))
    text_separed = [word for word in text_separed if word not in ignored_words]
    print(len(text_separed))
    text = " ".join(text_separed)
    # text = " ".join([word.lower() for word in text.split() if word.lower() not in words_not_relevant or word.lower() not in ignored_words])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )

    return wordcloud


def generate_word_frequencies_chart(df: pd.DataFrame, ignored_words=list(), color: str = "blue") -> alt.Chart:
    """
    Generate a bar chart of the 20 most frequent words from the cleaned text in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing 'cleaned_text' column.
        ignored_words (list): List of words to ignore in the frequency count.
        color (str): Color of the bars in the chart.

    Returns:
        alt.Chart: The generated bar chart of word frequencies.
    """
    # Clean special characters and separations
    df.loc[:, "cleaned_text"] = df["cleaned_text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

    # Join the filtered tokens back into a string
    text = " ".join(review for review in df["cleaned_text"])
    text_separated = text.lower().split(" ")
    text_separated = [word for word in text_separated if word not in words_not_relevant]
    text_separated = [word for word in text_separated if word not in ignored_words]
    text = " ".join(text_separated)

    # Generate word frequencies
    word_freq = Counter(text.split())
    word_freq_df = pd.DataFrame(word_freq.items(), columns=["word", "frequency"])

    # Filter out words that are not relevant
    word_freq_df = word_freq_df[~word_freq_df["word"].isin(words_not_relevant)]
    word_freq_df = word_freq_df[~word_freq_df["word"].isin(ignored_words)]
    
    # Get the 20 most frequent words
    total_words = word_freq_df['word'].count()
    word_freq_df = word_freq_df.nlargest(10, "frequency")
    
    # Create bar chart
    bar_chart = (
        alt.Chart(word_freq_df)
        .mark_bar(color=color)
        .encode(x="frequency:Q", y=alt.Y("word:N", sort="-x"))
        .properties(width=400, height=400)
    )

    return bar_chart, total_words


def generate_word2vec(df: pd.DataFrame, three_dimensional: bool = False):
    """
    Generate a Word2Vec model from the cleaned text in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing 'cleaned_text' column.
        three_dimensional (bool): Whether the analysis should be done in 3D.

    Returns:
        restaurant_coords (np.array): PCA-projected coordinates of the restaurants.
    """

    # if len(df['restaurant_id'].unique()) < 2:
    #     raise ValueError ("error" , "Veuillez sélectionner au moins deux restaurants.")

    # if not {'cleaned_text'}.issubset(df.columns):
    #     raise ValueError("Dataframe must contain 'cleaned_text' column.")

    df_reviews = df.copy()
    df_reviews["tokens"] = df_reviews["cleaned_text"].apply(
        lambda x: word_tokenize(x.lower())
    )


    # Entraîner le modèle Word2Vec
    model = Word2Vec(
        sentences=df_reviews["tokens"],  # df["tokens"],
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg = 0 # 1 = skip-gram, 0 = CBOW
    )

    # Prétraitement des avis
    # Fonction pour obtenir le vecteur moyen d'un avis
    def get_avg_vector(tokens):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(100)
        # return vectors
        return np.mean(vectors, axis=0)

    # Calculer les vecteurs moyens pour chaque restaurant
    df_reviews["avg_vector"] = df_reviews["tokens"].apply(get_avg_vector)

    df_reviews["contributions"] = df_reviews["contributions"].fillna(1)


    def weighted_avg_vector(vectors, weights):
        vectors = np.vstack(vectors)
        return np.average(vectors, axis=0, weights=weights)
    
    weighted_reviews = df_reviews.groupby("restaurant_id").agg(
        weighted_avg_vector=pd.NamedAgg(
            column="avg_vector",
            aggfunc=lambda x: weighted_avg_vector(x, df_reviews.loc[x.index, 'contributions'])
        )
    )


    # Agréger les vecteurs par restaurant
    # restaurant_vectors = (
    #     df_reviews.groupby("restaurant_id")
    #     .agg(
    #         {"avg_vector": lambda x: np.mean(list(x), axis=0), "restaurant_id": "first"}
    #     )
    #     .reset_index(drop=True)
    # )

    # restaurant_vectors = restaurant_vectors.drop_duplicates(subset="restaurant_id")
    # restaurant_vectors = restaurant_vectors.set_index("restaurant_id")
    # restaurant_vectors["restaurant_name"] = df_reviews.drop_duplicates(
    #     subset="restaurant_id"
    # ).set_index("restaurant_id")["restaurant_name"]
    # restaurant_vectors.reset_index(inplace=True)
    # restaurant_names = restaurant_vectors["restaurant_name"]

    weighted_reviews["restaurant_name"] = df_reviews.drop_duplicates(subset="restaurant_id").set_index("restaurant_id")["restaurant_name"]
    weighted_reviews["restaurant_type"] = df_reviews.drop_duplicates(subset="restaurant_id").set_index("restaurant_id")["restaurant_type"]
    weighted_reviews["restaurant_price"] = df_reviews.drop_duplicates(subset="restaurant_id").set_index("restaurant_id")["restaurant_price"]
    weighted_reviews.reset_index(inplace=True)
    # restaurant_info_supp = pd.DataFrame()
    # restaurant_info_supp["restaurant_name"] = weighted_reviews["restaurant_name"]
    # restaurant_info_supp["restaurant_type"] = weighted_reviews["restaurant_type"]
    # restaurant_info_supp["restaurant_price"] = weighted_reviews["restaurant_price"]

    restaurant_names = weighted_reviews["restaurant_name"]
    if three_dimensional:
        ncp = 3
    else:
        ncp = 2
    # Réduction de dimensionnalité avec ACP
    pca = PCA(n_components=ncp)
    restaurant_coords = pca.fit_transform(
        np.array(weighted_reviews["weighted_avg_vector"].tolist())
    )

    return restaurant_coords, restaurant_names


def generate_spider_plot(emotions_df):
    """
    Générer un spider plot interactif pour un restaurant spécifique avec Plotly.

    Args:
        emotions_df (pd.DataFrame): Moyennes des émotions par restaurant.

    Returns:
        go.Figure: Figure Plotly contenant le spider plot.
    """
    # Création de la figure
    fig = go.Figure()

    # Get values for emotions only
    emotion_cols = [col for col in emotions_df.columns if col != "restaurant_name"]
    emotions = emotion_cols + [emotion_cols[0]]

    # standardize emotions by dividing by the maximum value
    emotions_df[emotion_cols] = (
        emotions_df[emotion_cols] / emotions_df[emotion_cols].max().max()
    )

    # Add trace for each restaurant
    for restaurant_id in emotions_df.index:
        restaurant_name = emotions_df.loc[restaurant_id, "restaurant_name"]
        values = emotions_df.loc[restaurant_id, emotion_cols].values.flatten().tolist()
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=emotions,
                fill="toself",
                name=restaurant_name,
                opacity=0.5,
                legendgroup=restaurant_name
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparaison des émotions par restaurant",
        width=800,
        height=600,
        legend=dict(
            title="Légende",
            groupclick="toggleitem"
        )
    )

    return fig


def generate_sentiments_analysis(df: pd.DataFrame):
    """
    Analyze the sentiment of the reviews in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing 'cleaned_text' column.

    Returns:
        pd.DataFrame: The dataframe with an additional 'sentiment' column.
    """
    # Ajout d'une colonne "sentiment" avec la polarité des reviews
    df["sentiment"] = df["review_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    # La polarité est comprise entre -1 et 1 (négatif à positif)

    # reviews par restaurant
    df_par_resto = df.groupby("restaurant_id")["sentiment"].mean()
    note_moyenne = df.groupby("restaurant_id")["rating"].mean()

    df_par_resto = pd.DataFrame(df_par_resto)
    note_moyenne = pd.DataFrame(note_moyenne)

    notes_moyennes = pd.merge(df_par_resto, note_moyenne, on="restaurant_id")
    notes_moyennes = notes_moyennes.rename(
        columns={"sentiment": "sentiment_moyen", "rating": "note_moyenne"}
    )
    # merge with restaurants
    data = pd.merge(df, notes_moyennes, on="restaurant_id")

    # Scatter plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["note_moyenne"],
            y=data["sentiment_moyen"],
            mode="markers+text",
            marker=dict(size=10, color="blue", opacity=0.8),
            text=data["restaurant_name"],
            textposition="top center",
            hoverinfo="text",  # Afficher uniquement le nom au survol
        )
    )
    fig.update_layout(
        title="Relation entre le Sentiment des Avis et la Note Moyenne",
        xaxis_title="Note Moyenne",
        yaxis_title="Score de Sentiment Moyen",
        width=1000,
        height=600,
    )

    # Fonction pour extraire les scores d'émotions
    def extract_emotions(text):
        """
        Fonction qui extrait les scores d'émotions d'un texte.
        Pour chaque émotion, on calcule le score en fonction du nombre d'occurrences.
        """
        emotion_scores = NRCLex(text).raw_emotion_scores
        # Normaliser par le nombre total d'émotions détectées (optionnel)
        total = sum(emotion_scores.values())
        if total > 0:
            return {emotion: score / total for emotion, score in emotion_scores.items()}
        return emotion_scores

    # Appliquer la fonction sur la colonne "review_text" et créer un DataFrame d'émotions
    emotion_data = df["review_text"].apply(extract_emotions)
    emotion_df = pd.DataFrame(emotion_data.tolist())

    # Reset index to be able to concatenate
    df.reset_index(drop=True, inplace=True)
    emotion_df.reset_index(drop=True, inplace=True)

    # Ajouter les scores d'émotions au DataFrame "df"
    df = pd.concat([df, emotion_df], axis=1)

    # Calculer les moyennes des émotions pour chaque restaurant
    emotions_par_resto = df.groupby("restaurant_id")[emotion_df.columns].mean()

    # Before returning, add restaurant names to emotions_par_resto
    restaurant_names = (
        df[["restaurant_id", "restaurant_name"]]
        .drop_duplicates()
        .set_index("restaurant_id")
    )
    emotions_par_resto = emotions_par_resto.join(restaurant_names)

    return emotions_par_resto, fig

def get_coordinates(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            formatted_address = data['results'][0]['formatted_address']
            address_components = data['results'][0]['address_components']
            location = data['results'][0]['geometry']['location']
            latitud = location['lat']
            longitud = location['lng']
            zip_code = next(
                (component['long_name'] for component in address_components 
                 if "postal_code" in component['types']), None)
            country = next(
                (component['long_name'] for component in address_components 
                 if "country" in component['types']), None)
            ville = next(
                (component['long_name'] for component in address_components 
                 if "locality" in component['types']), None)
            return {
                "address": formatted_address,
                "latitude": latitud,
                "longitude": longitud,
                "zip_code": zip_code,
                "country": country,
                "ville": ville
            }
        else:
            return None
    else:
        return None
            
            
def generate_evolution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Generate a time-series chart showing the evolution of sentiment and rating over time.
    """
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
         df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Aggregating by month for smoother lines
    evolution = df.set_index('date').resample('M').agg({
        'sentiment': 'mean',
        'rating': 'mean',
        'review_text': 'count'
    }).reset_index()
    
    # Rename for tooltip
    evolution = evolution.rename(columns={'review_text': 'review_count'})

    # Create base chart
    base = alt.Chart(evolution).encode(x=alt.X('date:T', title='Date'))

    # Line for Rating (scaled to fit with sentiment if needed, or separate axes)
    # Using dual axis might be complex in simple altair, so let's normalize or just show sentiment
    # Let's show Sentiment on left, Rating on right? Or just two lines on different scales?
    # For simplicity and readability on a dashboard, let's stick to SENTIMENT evolution primarily, 
    # as that's the "Analytics" focus.
    
    line_sentiment = base.mark_line(color='#FF4B4B').encode(
        y=alt.Y('sentiment:Q', title='Sentiment Moyen'),
        tooltip=['date:T', alt.Tooltip('sentiment', format='.2f'), 'review_count']
    )
    
    points = base.mark_circle(color='#FF4B4B').encode(
        y='sentiment:Q',
        tooltip=['date:T', alt.Tooltip('sentiment', format='.2f'), 'review_count']
    )

    chart = (line_sentiment + points).properties(
        title="Évolution du Sentiment au Fil du Temps (Mensuel)",
        width=600,
        height=300
    ).interactive()

    return chart


def generate_rating_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Generate a bar chart showing the distribution of ratings (1-5 stars).
    """
    rating_counts = df['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['rating', 'count']

    chart = alt.Chart(rating_counts).mark_bar(color='#0068C9').encode(
        x=alt.X('rating:O', title='Note (Etoiles)'),
        y=alt.Y('count:Q', title='Nombre d\'avis'),
        tooltip=['rating', 'count']
    ).properties(
        title="Distribution des Notes",
        width=400,
        height=300
    )
    
    return chart


def generate_ngrams_chart(df: pd.DataFrame, n: int = 2) -> alt.Chart:
    """
    Generate a bar chart of the most frequent n-grams in the cleaned text.
    """
    text = " ".join(df["cleaned_text"].dropna())
    tokens = text.split()
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Count frequencies
    ngram_counts = Counter(n_grams)
    
    # Create DataFrame
    ngram_df = pd.DataFrame(ngram_counts.items(), columns=['ngram', 'count'])
    ngram_df['ngram_str'] = ngram_df['ngram'].apply(lambda x: ' '.join(x))
    
    # Get top 15
    ngram_df = ngram_df.nlargest(15, 'count')
    
    title_map = {2: "Bigrammes (2 mots)", 3: "Trigrammes (3 mots)"}
    
    chart = alt.Chart(ngram_df).mark_bar(color='#E15759').encode(
        x=alt.X('count:Q', title='Fréquence'),
        y=alt.Y('ngram_str:N', sort='-x', title='Expression'),
        tooltip=['ngram_str', 'count']
    ).properties(
        title=f"Expressions Fréquentes - {title_map.get(n, f'{n}-grams')}",
        width=400,
        height=300
    )
    
    return chart

def generate_pos_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Generate a chart showing top Adjectives and Nouns using NLTK POS tagging.
    """
    text = " ".join(df["cleaned_text"].dropna())
    tokens = word_tokenize(text) # Simple tokenization
    
    # Attempt POS tagging 
    tagged = pos_tag(tokens)
    
    # Filter for Adjectives (JJ) and Nouns (NN)
    adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    
    # Count top 10 of each
    top_adj = Counter(adjectives).most_common(10)
    top_noun = Counter(nouns).most_common(10)
    
    # Prepare DF
    adj_df = pd.DataFrame(top_adj, columns=['word', 'count'])
    adj_df['type'] = 'Adjectif'
    
    noun_df = pd.DataFrame(top_noun, columns=['word', 'count'])
    noun_df['type'] = 'Nom'
    
    combined_df = pd.concat([adj_df, noun_df])
    
    chart = alt.Chart(combined_df).mark_bar().encode(
        x=alt.X('count:Q', title='Fréquence'),
        y=alt.Y('word:N', sort='-x', title='Mot'),
        color=alt.Color('type:N', scale=alt.Scale(scheme='tableau10')),
        tooltip=['word', 'count', 'type']
    ).properties(
        title="Top Adjectifs et Noms (Approximation)",
        width=400,
        height=300
    )
    
    return chart
