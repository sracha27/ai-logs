"""
Logging simulation script by Kaushik
Generates mixed logs: HTTP successes, HTTP errors, and DB errors.
"""

import logging
import random
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='mixed_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Sample HTTP statuses and messages
http_codes = [
    (200, "OK"),
    (201, "Created"),
    (400, "Bad Request"),
    (401, "Unauthorized"),
    (403, "Forbidden"),
    (404, "Not Found"),
    (500, "Internal Server Error"),
    (502, "Bad Gateway"),
]

# Possible DB errors
db_errors = [
    "DBConnectionError: Could not connect to PostgreSQL server",
    "DBTimeoutError: Connection to database timed out",
    "DBAuthError: Invalid username or password",
    "DBPoolError: Connection pool exhausted"
]

def generate_log():
    roll = random.randint(1, 10)

    if roll <= 4:
        # 40% chance: HTTP 200 OK
        code, msg = 200, "OK"
        logging.debug(f"Request successful - HTTP {code} {msg}")
    elif roll <= 6:
        # 20% chance: DB error
        error = random.choice(db_errors)
        logging.error(f"Database error occurred - {error}")
    else:
        # 40% chance: random HTTP error (excluding 200 OK)
        code, msg = random.choice([c for c in http_codes if c[0] != 200])
        logging.warning(f"Request failed - HTTP {code} {msg}")

def main():
    try:
        while True:
            generate_log()
            time.sleep(2)  # Log every 2 seconds
    except KeyboardInterrupt:
        print("Logging stopped by user.")

if __name__ == "__main__":
    main()
