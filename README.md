# Scraper NLP TripAdvisor

This project demonstrates the application of Natural Language Processing (NLP) techniques to analyze restaurant reviews from TripAdvisor. It features a complete pipeline from data scraping to an interactive analytics dashboard and an AI-powered chat assistant.

## Features

-   **Data Scraping**: Automated scraping of restaurant reviews using BeautifulSoup.
-   **Sentiment Analysis**: LLM-based analysis of reviews to identify positive and negative points.
-   **Interactive Dashboard**: Visual analytics of review distribution, sentiment evolution, and key metrics using Streamlit.
-   **RAG Chatbot**: Chat with your data using Retrieval-Augmented Generation powered by Mistral AI.
-   **Geospatial Analysis**: Interactive map to explore restaurant locations and metrics.

## Architecture

A scraper (BeautifulSoup) gathers restaurant data, which is stored in a SQLite database. A Python backend serves this data to a Streamlit frontend. Mistral AI is integrated for advanced NLP tasks.

![Architecture Diagram](assets/img/architecture.png)

## Database

The database follows a star schema, separating locations, reviews, and restaurant details.

![UML](assets/img/nlp_sql_uml.png)

## Prerequisites

-   Python 3.10+
-   Docker & Docker Compose (optional for containerized deployment)
-   API Keys:
    -   [Mistral AI](https://console.mistral.ai/)
    -   [Google Maps Platform](https://developers.google.com/maps) (for map features)

## Configuration

Create a file named `.env` in the root directory with your API keys:

```sh
SQLITE_PATH=sql/tripadvisor.db
MISTRAL_API_KEY=your_mistral_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

## Installation & Setup

### Option 1: Docker (Recommended)

Run the application in a containerized environment:

```bash
docker-compose up --build -d
```

The app will be available at `http://localhost:8502`.

To stop the application:

```bash
docker-compose down
```

### Option 2: Local Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the environment:**
    -   **Mac/Linux:**
        ```bash
        source venv/bin/activate
        ```
    -   **Windows:**
        ```bash
        venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Collaborators

- **[@maxenceLIOGIER](https://github.com/maxenceLIOGIER)**
- **[@Cyr-CK](https://github.com/Cyr-CK)**
- **[@jdalfons](https://github.com/jdalfons)**