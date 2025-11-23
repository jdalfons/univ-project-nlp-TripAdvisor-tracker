CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id INTEGER PRIMARY KEY,
    restaurant_name VARCHAR(255),
    restaurant_url VARCHAR(255),
    restaurant_avg_review FLOAT,
    restaurant_total_reviews INT,
    restaurant_price VARCHAR(50),
    restaurant_type VARCHAR(255),
    restaurant_about TEXT
);

CREATE TABLE IF NOT EXISTS locations (
    location_id INTEGER PRIMARY KEY,
    restaurant_id INTEGER NOT NULL,
    address TEXT,
    ville TEXT,
    code_postal TEXT,
    latitude REAL,
    longitude REAL,
    country VARCHAR(255),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY,
    restaurant_id INTEGER NOT NULL,
    user_name TEXT,
    review_text TEXT,
    date DATE,
    contributions INTEGER,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);