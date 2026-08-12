from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By 
import itertools
import json
import time
import math

def scrape_page(driver):
    # Find all reviews
    review_entries = driver.find_elements(By.CLASS_NAME,"ReviewCard")
    
    # Extract text from reviews
    reviews = []

    for entry in review_entries:
        user = entry.find_element(By.CLASS_NAME,"ReviewerProfile__name").text

        try:
            rating_element = entry.find_element(By.CSS_SELECTOR, "span.RatingStars")
            rating = int(rating_element.get_attribute("aria-label").split()[1])
        except:
            rating = None

        comment = entry.find_element(By.CLASS_NAME,"ReviewText").text
        
        date = entry.find_element(By.XPATH,'.//span[contains(@class,"Text__body3")]/a').text
     
        try:
            likes = entry.find_element(By.XPATH,'.//span[contains(@class, "Button__labelItem") and contains(normalize-space(.), "like")]').text
        except:
            likes = 0
        
        reviews.append({
            "user": user,
            "rating": rating,
            "comment": comment,
            "date": date,
            "likes": likes
        })
    
    return reviews

def load_next_page(driver):
    # Click Load More Reviews button
    button = driver.find_element(By.CSS_SELECTOR,"[data-testid='loadMore']")
    button.click()

def write_to_file(reviews, filename):
    try:
        # Load existing reviews if the file already exists
        with open(filename, "r", encoding="utf-8") as final:
            existing_reviews = json.load(final)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_reviews = []

    # Add newly scraped reviews
    existing_reviews.extend(reviews)

    # Write everything back to the file
    with open(filename, "w", encoding="utf-8") as final:
        json.dump(
            existing_reviews,
            final,
            indent=2,
            default=lambda x: list(x) if isinstance(x, tuple) else str(x)
        )

    print(f"Data written to JSON. Total reviews: {len(existing_reviews)}")

OUTPUT_FILE = "goodreads_reviews.json"
TIME_SLEEP = 10

def scrape_reviews(NUM_PAGES, URL):
    print("[Scrapper]: Scrapping data from GoodReads")
    #  Setup the drive
    driver = webdriver.Firefox()
    driver.get(URL)

    # Wait for page to load
    time.sleep(TIME_SLEEP)

def scrape_reviews(NUM_PAGES, URL):
    print("[Scrapper]: Scrapping data from GoodReads")

    driver = webdriver.Firefox()
    driver.get(URL)

    time.sleep(TIME_SLEEP)

    total_pages = math.ceil(NUM_PAGES / 30)

    try:
        for page in range(total_pages):
            print(f"\n[Scrapper]: Starting page {page + 1} of {total_pages}")

            print("[Scrapper]: Scraping page...")
            reviews = scrape_page(driver)

            print(f"[Scrapper]: Found {len(reviews)} reviews")

            print("[Scrapper]: Saving reviews...")
            write_to_file(reviews, OUTPUT_FILE)

            print(f"[Scrapper]: Completed page {page + 1} of {total_pages}")

            if page < total_pages - 1:
                print("[Scrapper]: Loading next page...")
                load_next_page(driver)

                print("[Scrapper]: Waiting for page...")
                time.sleep(TIME_SLEEP)

    finally:
        print("[Scrapper]: Closing browser...")
        driver.quit()