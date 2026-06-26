from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
    with open(filename, "w") as final:
        json.dump(
            reviews,
            final,
            indent=2,
            default=lambda x: list(x) if isinstance(x, tuple) else str(x)
        )

    print(f"Data written to JSON. {len(reviews)} rows saved.")

OUTPUT_FILE = "goodreads_reviews.json"
TIME_SLEEP = 15

def scrape_reviews(directory_path, NUM_PAGES, URL):
    print("[Scrapper]: Scrapping data from GoodReads")
    #  Setup the drive
    driver = webdriver.Firefox()
    driver.get(URL)
    
    with open(OUTPUT_FILE, "w"):
        pass

    # Wait for page to load
    time.sleep(TIME_SLEEP)

    all_reviews = []

    for page in range(math.ceil(NUM_PAGES / 30)):
        reviews = scrape_page(driver)
        all_reviews.extend(reviews)

        if (page + 1) % 10 == 0:
            write_to_file(all_reviews, OUTPUT_FILE)

        load_next_page(driver)
        time.sleep(TIME_SLEEP)
    
    # Close the browser
    driver.quit()

    write_to_file(all_reviews, directory_path + OUTPUT_FILE)
    print("[Scrapper]: Scrapping complete.")