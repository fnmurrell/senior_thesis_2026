from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By 
import itertools
import json
import time

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
        json.dump(reviews, final, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else str(x))
    return print("Data written to JSON.")

NUM_PAGES = 10856 // 30
OUTPUT_FILE = "goodreads_reviews.json"
TIME_SLEEP = 5
URL = "https://www.goodreads.com/book/show/46787/reviews?reviewFilters=eyJhZnRlciI6Ik9UTXdOeXd4TlRRME16RTNPVFEyTWpReiJ9"

def scrape_reviews():
    print("[Scrapper]: Scrapping data from GoodReads")
    #  Setup the drive
    driver = webdriver.Firefox()
    driver.get(URL)

    # Wait for page to load
    time.sleep(TIME_SLEEP)

    # Create reviews list
    all_reviews = []
    
    # loop through review pages, scrape data, append, and click button for more reviews
    for page in range(0,NUM_PAGES):
        all_reviews.append(scrape_page(driver))
        load_next_page(driver)
        time.sleep(TIME_SLEEP)

    # Close the browser
    driver.quit()

    # Flatten the reviews list
    final_reviews = list(itertools.chain.from_iterable(all_reviews))

    # Generate final dataset
    write_to_file(final_reviews,OUTPUT_FILE)