import os
import platform
import sqlite3
import pandas as pd
from dotenv import load_dotenv

if platform.system() == "Windows":
    # Specify the path to your .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    # Load the .env file
    load_dotenv(dotenv_path)
else:
    load_dotenv()

# Get SQLite DB path from environment variable or default
DB_PATH = os.environ.get("SQLITE_PATH", "restaurants.db")

def get_db_connection():
    """
    Establish a connection to the SQLite database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    except sqlite3.Error as err:
        print(f"Database connection error: {err}")
        return None

def get_cursor():
    """
    Validate and obtain a cursor for database operations.
    Note: In SQLite, we usually use the connection object to create a cursor.
    This function now returns (connection, cursor) or just cursor depending on usage,
    but to keep compatibility with existing code structure, we'll return a cursor 
    that is bound to a connection. However, we need to be careful about closing connections.
    
    Better approach for this refactor: Helper functions create their own connection/cursor.
    But to minimize changes, let's try to adapt.
    """
    conn = get_db_connection()
    if conn:
        return conn.cursor()
    return None

def init_db():
    """
    Initialize the SQLite database with tables and data if it doesn't exist.
    """
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        print("Initializing database...")
        conn = get_db_connection()
        if conn is None:
            return
        
        try:
            cursor = conn.cursor()
            
            # List of SQL files to execute in order
            sql_files = [
                'sql/init.sql',
                'sql/set_restaurants.sql',
                'sql/set_locations.sql',
                'sql/set_reviews.sql'
            ]
            
            for sql_file in sql_files:
                print(f"Executing {sql_file}...")
                with open(sql_file, 'r') as f:
                    sql_script = f.read()
                    cursor.executescript(sql_script)
            
            conn.commit()
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")
        finally:
            conn.close()
    else:
        print("Database already exists.")

def get_all_reviews():
    """
    Fetch all reviews from the database.

    Returns:
        pd.DataFrame: DataFrame containing all reviews.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        reviews = conn.execute("SELECT * FROM reviews").fetchall()
        return pd.DataFrame([dict(review) for review in reviews])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        conn.close()


def get_all_restaurants():
    """
    Fetch all restaurants from the database.

    Returns:
        pd.DataFrame: DataFrame containing all restaurants.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        restaurants = conn.execute("SELECT * FROM restaurants").fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        conn.close()


def get_restaurant_by_type(restaurant_type):
    """
    Fetch restaurants by type from the database.

    Args:
        restaurant_type (str): The type of restaurant to fetch.

    Returns:
        pd.DataFrame: DataFrame containing restaurants of the specified type.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        restaurants = conn.execute(
            "SELECT * FROM restaurants WHERE restaurant_type = ?",
            (restaurant_type,)
        ).fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        conn.close()


def get_reviews_info_by_restaurant(restaurant_id):
    """
    Fetch summary of reviews for a specific restaurant by its ID.

    Args:
        restaurant_id (int): The ID of the restaurant.

    Returns:
        pd.DataFrame: DataFrame containing summary of reviews for the specified restaurant.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        review_summary = conn.execute(
            """
            SELECT 
                COUNT(*) AS review_count, 
                AVG(rating) AS average_rating, 
                MAX(date) AS last_comment_date, 
                MIN(date) AS first_comment_date 
            FROM reviews 
            WHERE restaurant_id = ?
            """,
            (int(restaurant_id),)
        ).fetchall()
        return pd.DataFrame([dict(summary) for summary in review_summary])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        conn.close()


def save_reviews_to_db(restaurant_id, reviews):
    """
    Save reviews to the database.

    Args:
        restaurant_id (int): The ID of the restaurant.
        reviews (list): List of reviews to save.
    """
    conn = get_db_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        for review in reviews:
            cursor.execute(
                """
                INSERT INTO reviews (restaurant_id, user_name, review_text, date, contributions, rating)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    restaurant_id, review['user_name'], review['review_text'],
                    review['date'], review['contributions'], review['rating']
                )
            )
        conn.commit()
    except sqlite3.Error as err:
        print(err)
    finally:
        conn.close()


def save_restaurant_to_db(restaurant_data):
    """
    Save restaurant data to the database.

    Args:
        restaurant_data (dict): Dictionary containing restaurant data.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        # Clean and format data
        restaurant_name = restaurant_data["restaurant_name"]
        restaurant_avg_review = float(restaurant_data["restaurant_avg_review"])
        restaurant_price = restaurant_data["restaurant_price"]
        restaurant_reviews = int(restaurant_data["restaurant_reviews"])
        restaurant_type_resto = restaurant_data["restaurant_type_resto"]
        restaurant_url = restaurant_data["restaurant_url"]

        address = restaurant_data["restauranta_address"]["address"]
        latitude = float(restaurant_data["restauranta_address"]["latitude"])
        longitude = float(restaurant_data["restauranta_address"]["longitude"])
        zip_code = restaurant_data["restauranta_address"]["zip_code"]
        country = restaurant_data["restauranta_address"]["country"]
        ville = restaurant_data["restauranta_address"]["ville"]

        # Insert restaurant data
        cursor.execute(
            """
            INSERT INTO restaurants (restaurant_name, restaurant_avg_review, restaurant_price, restaurant_total_reviews, restaurant_type, restaurant_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (restaurant_name, restaurant_avg_review, restaurant_price, restaurant_reviews, restaurant_type_resto, restaurant_url)
        )
        restaurant_id = cursor.lastrowid

        # Insert location data
        cursor.execute(
            """
            INSERT INTO locations (restaurant_id, address, latitude, longitude, code_postal, ville, country)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (restaurant_id, address, latitude, longitude, zip_code, ville, country)
        )

        conn.commit()
        return pd.DataFrame([{
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant_name,
            "restaurant_avg_review": restaurant_avg_review,
            "restaurant_price": restaurant_price,
            "restaurant_total_reviews": restaurant_reviews,
            "restaurant_type": restaurant_type_resto,
            "restaurant_url": restaurant_url
        }])
    except sqlite3.Error as err:
        print(err)
    finally:
        conn.close()

def delete_reviews_by_restaurant_id(restaurant_id):
    """
    Delete all reviews for a specific restaurant.

    Args:
        restaurant_id (int): The ID of the restaurant.
    """
    conn = get_db_connection()
    if conn is None:
        return
    try:
        conn.execute(
            "DELETE FROM reviews WHERE restaurant_id = ?",
            (restaurant_id,)
        )
        conn.commit()
    except sqlite3.Error as err:
        print(err)
    finally:
        conn.close()
        
        
def restaurant_exists(restaurant_url):
    """
    Check if a restaurant exists in the database by its URL.

    Args:
        restaurant_url (str): The URL of the restaurant.

    Returns:
        bool: True if the restaurant exists, False otherwise.
    """
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.execute(
            "SELECT 1 FROM restaurants WHERE restaurant_url = ?",
            (restaurant_url,)
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as err:
        print(err)
        return False
    finally:
        conn.close()

def get_downloaded_restaurants():
    """
    Fetch restaurants that have been downloaded.

    Returns:
        pd.DataFrame: DataFrame containing downloaded restaurants.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        restaurants = conn.execute("""
            SELECT r.restaurant_id,
                r.restaurant_name,
                r.restaurant_avg_review,
                r.restaurant_price,
                r.restaurant_type, 
                r.restaurant_total_reviews,
                r.restaurant_url,
                r.restaurant_about,
                l.address,
                l.latitude, 
                l.longitude,
                l.country,
                l.ville
            FROM restaurants r
            JOIN locations l ON r.restaurant_id = l.restaurant_id
            WHERE r.restaurant_id IN (SELECT restaurant_id FROM reviews)
        """).fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        conn.close()


def get_restaurant_by_id(restaurant_ids):
    """
    Fetch restaurants by their IDs.

    Args:
        restaurant_ids (list): List of restaurant IDs to fetch.

    Returns:
        pd.DataFrame: DataFrame containing restaurants with the specified IDs.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        if not restaurant_ids:
            return pd.DataFrame()

        # Chunk the IDs to avoid SQLite limit (usually 999 variables)
        chunk_size = 900
        all_restaurants = []
        
        for i in range(0, len(restaurant_ids), chunk_size):
            chunk = restaurant_ids[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            
            query = f"""
                SELECT 
                    r.restaurant_id,
                    r.restaurant_name,
                    r.restaurant_avg_review,
                    r.restaurant_type,
                    r.restaurant_price,
                    l.latitude,
                    l.longitude,
                    r2.rating,
                    r2.review_text,
                    r2.contributions,
                    r2.date
                FROM restaurants r
                JOIN locations l ON l.restaurant_id = r.restaurant_id
                JOIN reviews r2 ON r2.restaurant_id = r.restaurant_id 
                WHERE r.restaurant_id IN ({placeholders})
            """
            
            restaurants_chunk = conn.execute(query, chunk).fetchall()
            all_restaurants.extend(restaurants_chunk)
            
        return pd.DataFrame([dict(restaurant) for restaurant in all_restaurants])
    except sqlite3.Error as err:
        print(f"Error in get_restaurant_by_id: {err}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_reviews_one_restaurant(id):
    """
    Fetch reviews for a specific restaurant.

    Args:
        id (int): The ID of the restaurant.

    Returns:
        pd.DataFrame: DataFrame containing reviews for the specified restaurant.
    """
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        reviews = conn.execute(
            "SELECT * FROM reviews WHERE restaurant_id = ?",
            (int(id),)
        ).fetchall()
        return pd.DataFrame([dict(review) for review in reviews])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        conn.close()