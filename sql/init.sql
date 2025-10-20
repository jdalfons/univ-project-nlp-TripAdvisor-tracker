CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_name TEXT,
    restaurant_url TEXT,
    restaurant_avg_review REAL,
    restaurant_total_reviews INTEGER,
    restaurant_price TEXT,
    restaurant_type TEXT,
    restaurant_about TEXT
);

CREATE TABLE IF NOT EXISTS locations (
    location_id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id INTEGER NOT NULL,
    address TEXT,
    ville TEXT,
    code_postal TEXT,
    latitude REAL,
    longitude REAL,
    country TEXT,
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id INTEGER NOT NULL,
    user_name TEXT,
    review_text TEXT,
    date TEXT,
    contributions INTEGER,
    rating REAL CHECK (rating BETWEEN 1 AND 5),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);