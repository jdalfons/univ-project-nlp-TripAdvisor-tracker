import os
import platform
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

if platform.system() == "Windows":
    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path)
else:
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "sql" / "tripadvisor.db"


def resolve_sqlite_path() -> Path:
    """Resolve the SQLite database path from the environment or defaults."""
    raw_path = os.environ.get("SQLITE_PATH")
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
    else:
        candidate = DEFAULT_DB_PATH

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def run_sql_script(connection: sqlite3.Connection, script_path: Path) -> None:
    """Execute a SQL script if it exists."""
    if not script_path.exists():
        print(f"SQL script not found: {script_path}")
        return

    try:
        with script_path.open("r", encoding="utf-8") as sql_file:
            connection.executescript(sql_file.read())
        connection.commit()
    except sqlite3.Error as err:
        print(f"Error executing {script_path.name}: {err}")


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the SQLite database."""
    try:
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as err:
        print(f"Error checking table {table_name}: {err}")
        return False


def seed_table(connection: sqlite3.Connection, table_name: str, script_name: str) -> None:
    """Populate a table from a SQL script if it is empty."""
    if not table_exists(connection, table_name):
        return

    try:
        cursor = connection.execute(f"SELECT COUNT(*) FROM {table_name}")
        row = cursor.fetchone()
        count = row[0] if row else 0
    except sqlite3.Error as err:
        print(f"Error reading {table_name} row count: {err}")
        return

    if count == 0:
        scripts_dir = BASE_DIR / "sql"
        run_sql_script(connection, scripts_dir / script_name)


def initialize_database(connection: sqlite3.Connection) -> None:
    """Ensure the SQLite database schema and demo data are available."""
    try:
        connection.execute("PRAGMA foreign_keys = ON;")
    except sqlite3.Error as err:
        print(f"Error enabling foreign keys: {err}")

    scripts_dir = BASE_DIR / "sql"
    run_sql_script(connection, scripts_dir / "init.sql")

    # Seed demo data if the tables are empty
    seed_table(connection, "restaurants", "set_restaurants.sql")
    seed_table(connection, "locations", "set_locations.sql")
    seed_table(connection, "reviews", "set_reviews.sql")


try:
    DB_PATH = resolve_sqlite_path()
    db = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
    )
    db.row_factory = sqlite3.Row
    initialize_database(db)
except sqlite3.Error as err:
    print(f"Database connection error: {err}")
    db = None


def get_cursor():
    """
    Validate and obtain a cursor for database operations.

    Returns:
        cursor: A database cursor.
    """
    if db is None:
        return None

    try:
        return db.cursor()
    except sqlite3.Error as err:
        print(f"Error obtaining cursor: {err}")
        return None


def get_all_reviews():
    """
    Fetch all reviews from the database.

    Returns:
        pd.DataFrame: DataFrame containing all reviews.
    """
    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        cursor.execute("SELECT * FROM reviews")
        reviews = cursor.fetchall()
        return pd.DataFrame([dict(review) for review in reviews])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()


def get_all_restaurants():
    """
    Fetch all restaurants from the database.

    Returns:
        pd.DataFrame: DataFrame containing all restaurants.
    """
    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        cursor.execute("SELECT * FROM restaurants")
        restaurants = cursor.fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()


def get_restaurant_by_type(restaurant_type):
    """
    Fetch restaurants by type from the database.

    Args:
        restaurant_type (str): The type of restaurant to fetch.

    Returns:
        pd.DataFrame: DataFrame containing restaurants of the specified type.
    """
    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        cursor.execute(
            "SELECT * FROM restaurants WHERE restaurant_type = ?",
            (restaurant_type,),
        )
        restaurants = cursor.fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()


def get_reviews_info_by_restaurant(restaurant_id):
    """
    Fetch summary of reviews for a specific restaurant by its ID.

    Args:
        restaurant_id (int): The ID of the restaurant.

    Returns:
        pd.DataFrame: DataFrame containing summary of reviews for the specified restaurant.
    """
    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS review_count,
                AVG(rating) AS average_rating,
                MAX(date) AS last_comment_date,
                MIN(date) AS first_comment_date
            FROM reviews
            WHERE restaurant_id = ?
            """,
            (int(restaurant_id),),
        )
        review_summary = cursor.fetchall()
        return pd.DataFrame([dict(summary) for summary in review_summary])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()


def save_reviews_to_db(restaurant_id, reviews):
    """
    Save reviews to the database.

    Args:
        restaurant_id (int): The ID of the restaurant.
        reviews (list): List of reviews to save.
    """
    if not reviews:
        return

    cursor = get_cursor()
    if cursor is None:
        return

    try:
        cursor.executemany(
            """
            INSERT INTO reviews (restaurant_id, user_name, review_text, date, contributions, rating)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    restaurant_id,
                    review.get("user_name"),
                    review.get("review_text"),
                    review.get("date"),
                    review.get("contributions"),
                    review.get("rating"),
                )
                for review in reviews
            ],
        )
        db.commit()
    except sqlite3.Error as err:
        print(err)
    finally:
        cursor.close()


def _to_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def save_restaurant_to_db(restaurant_data):
    """
    Save restaurant data to the database.

    Args:
        restaurant_data (dict): Dictionary containing restaurant data.
    """
    cursor = get_cursor()
    if cursor is None:
        return

    try:
        restaurant_name = restaurant_data.get("restaurant_name")
        restaurant_avg_review = _to_float(restaurant_data.get("restaurant_avg_review"))
        restaurant_price = restaurant_data.get("restaurant_price")
        restaurant_reviews = _to_int(restaurant_data.get("restaurant_reviews"))
        restaurant_type_resto = restaurant_data.get("restaurant_type_resto")
        restaurant_url = restaurant_data.get("restaurant_url")
        restaurant_about = restaurant_data.get("restaurant_about")

        address_info = restaurant_data.get("restauranta_address") or {}
        address = address_info.get("address")
        latitude = _to_float(address_info.get("latitude"))
        longitude = _to_float(address_info.get("longitude"))
        zip_code = address_info.get("zip_code")
        country = address_info.get("country")
        ville = address_info.get("ville")

        cursor.execute(
            """
            INSERT INTO restaurants (
                restaurant_name,
                restaurant_avg_review,
                restaurant_price,
                restaurant_total_reviews,
                restaurant_type,
                restaurant_url,
                restaurant_about
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                restaurant_name,
                restaurant_avg_review,
                restaurant_price,
                restaurant_reviews,
                restaurant_type_resto,
                restaurant_url,
                restaurant_about,
            ),
        )
        restaurant_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO locations (restaurant_id, address, latitude, longitude, code_postal, ville, country)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                restaurant_id,
                address,
                latitude,
                longitude,
                zip_code,
                ville,
                country,
            ),
        )

        db.commit()

        return pd.DataFrame([
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "restaurant_avg_review": restaurant_avg_review,
                "restaurant_price": restaurant_price,
                "restaurant_total_reviews": restaurant_reviews,
                "restaurant_type": restaurant_type_resto,
                "restaurant_url": restaurant_url,
            }
        ])
    except sqlite3.Error as err:
        print(err)
    finally:
        cursor.close()


def delete_reviews_by_restaurant_id(restaurant_id):
    """
    Delete all reviews for a specific restaurant.

    Args:
        restaurant_id (int): The ID of the restaurant.
    """
    cursor = get_cursor()
    if cursor is None:
        return

    try:
        cursor.execute(
            "DELETE FROM reviews WHERE restaurant_id = ?",
            (restaurant_id,),
        )
        db.commit()
    except sqlite3.Error as err:
        print(err)
    finally:
        cursor.close()


def restaurant_exists(restaurant_url):
    """
    Check if a restaurant exists in the database by its URL.

    Args:
        restaurant_url (str): The URL of the restaurant.

    Returns:
        bool: True if the restaurant exists, False otherwise.
    """
    cursor = get_cursor()
    if cursor is None:
        return False

    try:
        cursor.execute(
            "SELECT 1 FROM restaurants WHERE restaurant_url = ?",
            (restaurant_url,),
        )
        return cursor.fetchone() is not None
    except sqlite3.Error as err:
        print(err)
        return False
    finally:
        cursor.close()


def get_downloaded_restaurants():
    """
    Fetch restaurants that have been downloaded.

    Returns:
        pd.DataFrame: DataFrame containing downloaded restaurants.
    """
    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        cursor.execute(
            """
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
            """
        )
        restaurants = cursor.fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()


def get_restaurant_by_id(restaurant_ids):
    """
    Fetch restaurants by their IDs.

    Args:
        restaurant_ids (list): List of restaurant IDs to fetch.

    Returns:
        pd.DataFrame: DataFrame containing restaurants with the specified IDs.
    """
    if not restaurant_ids:
        return pd.DataFrame()

    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        placeholders = ",".join(["?"] * len(restaurant_ids))
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
                r2.contributions
            FROM restaurants r
            JOIN locations l ON l.restaurant_id = r.restaurant_id
            JOIN reviews r2 ON r2.restaurant_id = r.restaurant_id
            WHERE r.restaurant_id IN ({placeholders})
        """
        cursor.execute(query, restaurant_ids)
        restaurants = cursor.fetchall()
        return pd.DataFrame([dict(restaurant) for restaurant in restaurants])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()


def get_reviews_one_restaurant(id):
    """
    Fetch reviews for a specific restaurant.

    Args:
        id (int): The ID of the restaurant.

    Returns:
        pd.DataFrame: DataFrame containing reviews for the specified restaurant.
    """
    cursor = get_cursor()
    if cursor is None:
        return pd.DataFrame()

    try:
        cursor.execute(
            "SELECT * FROM reviews WHERE restaurant_id = ?",
            (int(id),),
        )
        reviews = cursor.fetchall()
        return pd.DataFrame([dict(review) for review in reviews])
    except sqlite3.Error as err:
        print(err)
        return pd.DataFrame()
    finally:
        cursor.close()
