from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import itertools
import time

def scrape_page(driver):
    # Find all reviews
    review_entries = driver.find_elements(By.CLASS_NAME,"ReviewCard")
    
    # Extract text from reviews
    reviews = []

    for entry in review_entries:
        user = entry.find_element(By.CLASS_NAME,"ReviewerProfile__name").text

        rating_element = entry.find_element(By.CSS_SELECTOR, "span.RatingStars")
        rating = int(rating_element.get_attribute("aria-label").split()[1])

        comment = entry.find_element(By.CLASS_NAME,"ReviewText").text
        
        date = entry.find_element(By.XPATH,'.//span[contains(@class,"Text__body3")]/a').text
     
        likes = entry.find_element(By.XPATH,'.//span[contains(@class, "Button__labelItem") and contains(normalize-space(.), "like")]').text
        
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

def write_to_file(reviews,filename):
    pass

NUM_PAGES = 5 #10850 / 30
OUTPUT_FILE = "goodreads_reviews.json"
TIME_SLEEP = 3
URL = "https://www.goodreads.com/book/show/46787/reviews?reviewFilters=eyJhZnRlciI6Ik9UTXdOeXd4TlRRME16RTNPVFEyTWpReiJ9"

def main():
    #  Setup the drive
    driver = webdriver.Firefox()
    driver.get(URL)

    # Wait for page to load
    time.sleep(TIME_SLEEP)

    # Create reviews list
    all_reviews = []
    
    # loop through review pages, scrape data, append, and click button for more reviews
    for page in range(0,NUM_PAGES):
        all_reviews.extend(scrape_page(driver))
        load_next_page(driver)
        time.sleep(TIME_SLEEP)

    # Close the browser
    driver.quit()

    # Flatten the reviews list
    final_reviews = list(itertools.chain.from_iterable(all_reviews))
    print(final_reviews)

    # Generate final dataset
    write_to_file(final_reviews,OUTPUT_FILE)

if __name__ == "__main__":
    main()