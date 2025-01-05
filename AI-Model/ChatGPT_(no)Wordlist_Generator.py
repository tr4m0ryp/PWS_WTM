from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

chrome_options = Options()
chrome_options.add_argument("--start-maximized") 
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

service = Service('C:/Downloads/chromedriver.exe')
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    driver.get("https://chat.openai.com/")
    time.sleep(5)
    try:
        text_box = driver.find_element(By.TAG_NAME, "textarea")
        text_box.click()
        time.sleep(2)
        text_box.send_keys("Hallo, test test test")
        text_box.send_keys(Keys.RETURN)
        print("Prompt succesvol ingevoerd!")
    except Exception as e:
        print("Kon het tekstveld niet vinden of er is een ander probeleem: ", e)
    
    time.sleep(10)
