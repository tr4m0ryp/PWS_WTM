import os
import aiohttp
import asyncio
import requests
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from aiohttp import ClientSession, TCPConnector
import ssl
import logging
import aiofiles 
import queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_gathering.log"),
        logging.StreamHandler()
    ]
)

API_KEYS = ["c5469d1f63a549f3a675a528beaa1de7", "e089e406032b45f287479d812a116464"]
API_KEYS_lock = threading.Lock()
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

input_excel = "D:/ProfielWerkstuk/PWS_Main_Files/DATA_gathering/Wordlist_Generater/Generated_Item_List/shuffled_WordList_Mark7.xlsx"
output_path = "C:/ProfielWerkstuk/PWS_Bing"
output_csv_name = "C:/ProfielWerkstuk/PWS_Bing/Train_data_AI_Model.csv"
url_cache_file = "C:/ProfielWerkstuk/PWS_Bing/url_cache.txt"
checkpoint_file = "C:/ProfielWerkstuk/PWS_Bing/checkpoint.txt"
failed_words_file = "C:/ProfielWerkstuk/PWS_Bing/failed_words.txt"

MAX_CONCURRENT_REQUESTS = 20
MAX_WORKERS = 25
CHECKPOINT_INTERVAL = 10

retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


df_results = pd.DataFrame(columns=["Item", "Afbeelding pad", "Category"])

df_results_lock = asyncio.Lock()
checkpoint_lock = asyncio.Lock()
cache_lock = asyncio.Lock()
failed_words_lock = asyncio.Lock()

pause_on_401_event = asyncio.Event()
pause_on_401_event.set()

ssl_context = ssl.create_default_context()
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

def load_url_cache():
    if os.path.exists(url_cache_file):
        with open(url_cache_file, "r") as f:
            return set(f.read().splitlines())
    return set()

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
    return f"Single {item} made of {category} isolated on a white background"

async def exponential_backoff_async(attempt):
    wait_time = min(2 ** attempt, 300)
    logging.warning(f"[BACKOFF] Wachten voor {wait_time} seconden vanwege fout...")
    await asyncio.sleep(wait_time)

async def log_failed_item(item, category, reason, found_images=0):
    async with failed_words_lock:
        async with aiofiles.open(failed_words_file, "a") as file:
            await file.write(f"{item} (Categorie: {category}) - Reden: {reason} - Gevonden afbeeldingen: {found_images}\n")

def search_images(query, min_images=200):
    image_urls = []
    start = 0
    current_api_key_index = 0
    attempts = 0

    logging.info(f"Start zoeken voor: '{query}'")

    while len(image_urls) < min_images:
        with API_KEYS_lock:
            if not API_KEYS:
                logging.error("Geen API-sleutels beschikbaar. Voeg een nieuwe sleutel toe met 'add key [new_key]'.")
                pause_on_401_event.clear()
                raise UnauthorizedError("No API keys available")
            api_key = API_KEYS[current_api_key_index]

        search_url = f"{BING_ENDPOINT}?q={query}&count=50&offset={start}&mkt=en-US&safeSearch=Moderate"
        headers = {"Ocp-Apim-Subscription-Key": api_key}

        try:
            response = session.get(search_url, headers=headers, timeout=10)

            if response.status_code == 401:
                logging.error("401 error: Script gepauzeerd voor API-sleutel update")
                pause_on_401_event.clear() 
                raise UnauthorizedError("401 Unauthorized Error")

            if response.status_code == 429:
                logging.warning("Rate-limiet overschreden, API-sleutel wisselen...")
                with API_KEYS_lock:
                    current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)

                if current_api_key_index == 0:
                    asyncio.run(exponential_backoff_async(attempts))
                    attempts += 1
                continue

            response.raise_for_status()
            data = response.json()

            if 'value' in data:
                images = data['value']
                new_urls = [img['contentUrl'] for img in images]
                image_urls.extend(new_urls)
                logging.info(f"{len(new_urls)} nieuwe afbeeldingen gevonden (Totaal: {len(image_urls)})")
            else:
                logging.info(f"Geen verdere resultaten voor '{query}'")
                break

        except requests.exceptions.RequestException as e:
            asyncio.run(exponential_backoff_async(attempts))
            attempts += 1
            continue

        start += 50

    return image_urls[:min_images]

async def fetch_image(session, url, save_path, item, category, index, i):
    async with semaphore:
        try:
            if url in url_cache:
                return None 

            async with session.get(url, ssl=ssl_context) as response:
                if 'image' not in response.headers.get('Content-Type', ''):
                    return None

                content = await response.read()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(content)

                async with cache_lock:
                    url_cache.add(url)
                    async with aiofiles.open(url_cache_file, "a") as cache_file:
                        await cache_file.write(f"{url}\n")

                normalized_path = os.path.normpath(save_path)

                async with df_results_lock:
                    df_results.loc[len(df_results)] = [item, normalized_path, category]

                return save_path
        except Exception as e:
            return None

async def download_images_concurrently(image_urls, item, category, index, output_path):
    category_folder = os.path.join(output_path, category.replace(' ', '_'))
    os.makedirs(category_folder, exist_ok=True)

    connector = TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ssl=ssl_context)
    async with ClientSession(connector=connector) as session_aiohttp:
        tasks = [
            fetch_image(
                session_aiohttp, 
                url, 
                os.path.join(category_folder, f"{item.replace(' ', '_')}_{index}_{i}.png"),
                item,
                category,
                index,
                i
            )
            for i, url in enumerate(image_urls)
        ]
        await asyncio.gather(*tasks)

async def process_item(item, category, index, total_items, output_path, min_images_per_item=200, progress=None):
    if pd.isnull(item) or pd.isnull(category) or str(item).strip() == "" or str(category).strip() == "":
        await log_failed_item(item, category, "Lege of ongeldige waarde")
        logging.warning(f"Item {index + 1}/{total_items} overgeslagen vanwege lege of ongeldige waarde")
        if progress is not None:
            progress['completed_items'] += 1
        return

    search_term = refine_search_term(item, category)
    logging.info(f"Verwerken van item {index + 1}/{total_items}: {search_term}")

    while True:
        try:
            image_urls = search_images(search_term, min_images=min_images_per_item)
            break 
        except UnauthorizedError:
            logging.info("Wachten op nieuwe API-sleutel...")
            await pause_on_401_event.wait() 

    if len(image_urls) < min_images_per_item:
        logging.warning(f"Niet genoeg afbeeldingen gevonden voor '{item}' (Categorie: {category})")
        await log_failed_item(item, category, "Niet genoeg afbeeldingen gevonden", len(image_urls))
        if progress is not None:
            progress['completed_items'] += 1
        return

    await download_images_concurrently(image_urls, item, category, index, output_path)

    async with checkpoint_lock:
        save_checkpoint(item)

    if progress is not None:
        progress['completed_items'] += 1

def save_results_to_csv():
    df_results.to_csv(output_csv_name, index=False)
    logging.info(f"Resultaten tussentijds opgeslagen naar {output_csv_name}")

def listen_for_pause_or_resume(pause_event):
    global MAX_WORKERS
    while True:
        command = input("\nTyp 'pauze' om te pauzeren, 'hervat' om verder te gaan, 'resume401' om door te gaan na 401-fout, 'add key [new_key]' om een nieuwe API-sleutel toe te voegen, of 'set workers [n]' om het aantal workers aan te passen: ").strip().lower()
        if command == 'pauze':
            pause_event.clear()
            logging.info("*** Script gepauzeerd ***")
        elif command == 'hervat':
            pause_event.set()
            logging.info("*** Script hervat ***")
        elif command == 'resume401':
            pause_on_401_event.set()
            logging.info("*** Script hervat na 401-fout ***")
        elif command.startswith('add key'):
            try:
                new_key = command.split(' ', 2)[2]
                with API_KEYS_lock:
                    API_KEYS.append(new_key)
                logging.info(f"*** Nieuwe API-sleutel toegevoegd. Totaal sleutels: {len(API_KEYS)} ***")
                pause_on_401_event.set()
            except IndexError:
                logging.error("[FOUT] Onjuiste opdracht. Gebruik: 'add key [new_key]'.")
        elif command.startswith('set workers'):
            try:
                new_workers = int(command.split()[2])
                if new_workers > 0:
                    MAX_WORKERS = new_workers
                    logging.info(f"*** Aantal workers aangepast naar {MAX_WORKERS} ***")
                else:
                    logging.error("[FOUT] Ongeldig aantal workers.")
            except (IndexError, ValueError):
                logging.error("[FOUT] Onjuiste opdracht. Gebruik: 'set workers [n]'.")
        else:
            logging.warning("Onbekende opdracht.")

async def log_estimated_time(total_items, progress, start_time):
    while progress['completed_items'] < total_items:
        await asyncio.sleep(60)
        elapsed_time = time.time() - start_time
        completed = progress['completed_items']
        if completed > 0:
            average_time_per_item = elapsed_time / completed
            remaining_items = total_items - completed
            estimated_remaining = remaining_items * average_time_per_item
            estimated_remaining_formatted = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining))
            logging.info(f"Voortgang: {completed}/{total_items} items verwerkt. Geschatte resterende tijd: {estimated_remaining_formatted}")
    logging.info("Alle items zijn verwerkt.")

async def run_image_collection(input_excel, output_path, pause_event, min_images_per_item=200):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_excel(input_excel, engine='openpyxl')
    total_items = len(df)

    checkpoint_item = load_checkpoint()
    start_processing = False if checkpoint_item else True

    progress = {'completed_items': 0}
    start_time = time.time()

    estimated_time_task = asyncio.create_task(log_estimated_time(total_items, progress, start_time))

    tasks = []
    for index, row in df.iterrows():
        item = row['Item']
        category = row['Category']

        if not start_processing:
            if item == checkpoint_item:
                start_processing = True
            else:
                continue 

        await pause_event.wait()        
        await pause_on_401_event.wait()

        task = asyncio.create_task(process_item(item, category, index, total_items, output_path, min_images_per_item, progress))
        tasks.append(task)

        if (progress['completed_items'] + len(tasks)) % 50 == 0:
            await asyncio.gather(*tasks)
            tasks.clear()
            save_results_to_csv() 

        if len(tasks) >= MAX_WORKERS:
            await asyncio.gather(*tasks)
            tasks.clear()

    if tasks:
        await asyncio.gather(*tasks)
    save_results_to_csv()

    await estimated_time_task


if __name__ == "__main__":
    pause_event = asyncio.Event()
    pause_event.set()

    listener_thread = threading.Thread(target=listen_for_pause_or_resume, args=(pause_event,), daemon=True)
    listener_thread.start()

    try:
        asyncio.run(run_image_collection(input_excel, output_path, pause_event, min_images_per_item=200))
    except Exception as e:
        logging.critical(f"Onverwachte fout: {e}")
