
API_KEY = "AIzaSyBvVL8cqOZwzwW9qScEb1MVKMM1u6ZZOkU"
CX = "978082a0a86dd420d"
import os
import aiohttp
import asyncio
import requests
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime
from io import BytesIO
from PIL import Image as PILImage
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

input_excel = "C:/Users/Moussa (CCNV)/Downloads/PWS/DATA_gathering/Wordlist_Generater/Generated_Item_List/shuffled_WordList_Mark7.xlsx"
output_path = "D:/PWS/"
output_csv_name = "D:/PWS/Train_data_AI_Model.csv"
checkpoint_file = "D:/PWS/checkpoint.txt"
url_cache_file = "D:/PWS/url_cache.txt"
failed_words_file = "D:/PWS/failed_words.txt"

MAX_CONCURRENT_REQUESTS = 10  
MAX_WORKERS = 4  

CHECKPOINT_INTERVAL = 10

retry_strategy = Retry(
    total=3,  
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],  
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def load_url_cache():
    if os.path.exists(url_cache_file):
        with open(url_cache_file, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_url_to_cache(url):
    with open(url_cache_file, "a") as f:
        f.write(f"{url}\n")

url_cache = load_url_cache()

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as file:
            return file.readline().strip()
    return None

def save_checkpoint(item):
    with open(checkpoint_file, "w") as file:
        file.write(item)

def refine_search_term(item, category):
    return f"Single {item} made of {category} isolated with a white backgroun"

def wait_until_next_day():
    now = datetime.now()
    next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    wait_time = (next_day - now).total_seconds()
    print(f"Dagelijkse API-limiet bereikt. Wachten tot middernacht... ({int(wait_time // 3600)} uur en {int((wait_time % 3600) // 60)} minuten)")
    
    while wait_time > 0:
        mins, secs = divmod(wait_time, 60)
        timer = f"{int(mins):02d}:{int(secs):02d}"
        print(f"\rWachttijd tot de volgende dag: {timer}", end="")
        time.sleep(1)
        wait_time -= 1
    print("\nHet is nu de volgende dag, doorgaan met het proces...")

def search_images(query, min_images=200):
    image_urls = []
    start = 1

    while len(image_urls) < min_images:
        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&searchType=image&num=10&start={start}"
        try:
            response = session.get(search_url)
            if response.status_code == 429:
                error_json = response.json()
                if 'quota exceeded' in error_json.get('error', {}).get('message', '').lower():
                    print("Dagelijkse API-limiet bereikt. Wachten tot de volgende dag...")
                    wait_until_next_day()
                    continue
                else:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        print(f"Rate-limiet overschreden! Wachten voor {retry_after} seconden...")
                        time.sleep(int(retry_after))
                    else:
                        print("Rate-limiet overschreden! Wachten voor 10 minuten...")
                        time.sleep(600)
                continue

            response.raise_for_status()  
            data = response.json()

            if 'items' in data:
                images = data['items']
                image_urls += [img['link'] for img in images]
                print(f"Gevonden {len(images)} nieuwe afbeeldingen voor zoekopdracht: {query} (Totaal verzameld: {len(image_urls)})")
            else:
                print(f"Geen verdere resultaten voor {query}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Fout tijdens het zoeken naar afbeeldingen voor {query}: {e}")
            break

        start += 10

    return image_urls[:min_images]

async def fetch_image(session, url, save_path):
    async with semaphore:
        try:
            if url in url_cache:
                return False

            async with session.get(url) as response:
                if 'image' not in response.headers['Content-Type']:
                    return False
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
                save_url_to_cache(url)
                return True
        except Exception as e:
            print(f"Fout bij het downloaden van afbeelding van {url}: {e}")
            return False

async def download_images_concurrently_async(image_urls, item, category, index, output_path):
    category_folder = os.path.join(output_path, category.replace(' ', '_'))
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_image(session, url, os.path.join(category_folder, f"{item.replace(' ', '_')}_{index}_{i}.png"))
            for i, url in enumerate(image_urls)
        ]
        await asyncio.gather(*tasks)

async def process_item(row, index, total_items, output_path, min_images_per_item=200):
    item = row['Item']
    category = row['Category']
    
    search_term = refine_search_term(item, category)
    print(f"Zoeken naar afbeeldingen voor: {search_term}")

    image_urls = search_images(search_term, min_images=min_images_per_item)
    if len(image_urls) < min_images_per_item:
        print(f"Niet genoeg afbeeldingen gevonden voor {item} (Categorie: {category})")
        return

    await download_images_concurrently_async(image_urls, item, category, index, output_path)

def listen_for_pause_or_resume(pause_event):
    global MAX_WORKERS  
    while True:
        command = input("\nTyp 'pauze' om te pauzeren, 'hervat' om verder te gaan, of 'set workers [n]' om het aantal workers aan te passen: ").strip().lower()
        if command == 'pauze':
            pause_event.clear()
            print("\n*** Script is gepauzeerd ***")
        elif command == 'hervat':
            pause_event.set()
            print("\n*** Script gaat verder... ***")
        elif command.startswith('set workers'):
            try:
                new_workers = int(command.split()[2])
                if new_workers > 0:
                    MAX_WORKERS = new_workers
                    print(f"\n*** Aantal workers aangepast naar {MAX_WORKERS} ***")
                else:
                    print("Ongeldig aantal workers. Gebruik een getal groter dan 0.")
            except (IndexError, ValueError):
                print("Onjuiste opdracht. Gebruik: 'set workers [n]' om het aantal workers in te stellen.")

async def run_image_collection(input_excel, output_path, output_csv_name, pause_event, min_images_per_item=200):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = pd.read_excel(input_excel)
    total_items = len(df)

    tasks = []
    for index, row in df.iterrows():
        pause_event.wait()  
        task = process_item(row, index, total_items, output_path, min_images_per_item)
        tasks.append(task)

        if len(tasks) >= MAX_WORKERS:
            await asyncio.gather(*tasks)
            tasks.clear()

    if tasks:
        await asyncio.gather(*tasks)

pause_event = threading.Event()
pause_event.set()

listener_thread = threading.Thread(target=listen_for_pause_or_resume, args=(pause_event,))
listener_thread.daemon = True
listener_thread.start()

asyncio.run(run_image_collection(input_excel, output_path, output_csv_name, pause_event, min_images_per_item=200))
