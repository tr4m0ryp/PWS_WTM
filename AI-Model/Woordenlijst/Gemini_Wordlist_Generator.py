import google.generativeai as genai
import pandas as pd
import os
import time
import re

api_sleutels = [
    "AIzaSyAr9ri1n7nbKals46vq-H3KWPyi72Ey3n8", #really? leaving your api-keys here?!; there aren't valid anymore ;)
    "AIzaSyAoR1zfRjrNGCSG9afFeAR2ncqvljqJVF0",  #really? leaving your api-keys here?!; there aren't valid anymore ;)
    "AIzaSyCKEIweMN_O_i8C8OWW2eZAnPqwy4KNWHA",#really? leaving your api-keys here?!; there aren't valid anymore ;)
    "AIzaSyB6O7sFcq9Wq4KjV291-upDloCt0dUMrJk",#really? leaving your api-keys here?!; there aren't valid anymore ;)
    "AIzaSyAwUkvZaCe9i-H0MBq8aoUMEoMguxEh4FY"#really? leaving your api-keys here?!; there aren't valid anymore ;)
]

def configureer_api(api_sleutel):
    genai.configure(api_key=api_sleutel)

huidige_api_sleutel_index = 0
configureer_api(api_sleutels[huidige_api_sleutel_index])

categorieën = {
    "Plastic": [
        "plastic bottle", "plastic bag", "plastic toy", "plastic cup", "plastic straw",
        "plastic container", "plastic wrap", "plastic cutlery", "plastic packaging",
        "PET bottle", "HDPE container", "plastic lid"
    ],
    "Paper and Cardboard": ["newspaper", "cardboard box", "paper bag", "brochure", "notebook"],
    "Metal": ["soda can", "beer can", "aluminum foil", "metal cap", "nail", "screw"],
    "Glass": ["glass bottle", "glass jar", "broken glass", "mirror", "window glass"],
    "Organic Waste": ["banana peel", "apple core", "mandarin peel", "coffee grounds", "tea bag", "leaves"],
    "Textile": ["old jeans", "worn t-shirt", "broken sock", "old towel", "cushion cover"],
    "Wood": ["wooden spoon", "broken chair", "branches", "wooden toy", "wooden plank"],
    "Electronics": ["old mobile phone", "battery", "laptop charger", "old keyboard", "broken mouse"]
}

def maak_item_tekst_schoon(tekst):
    tekst = re.sub(r"^[\-\*\s]+", "", tekst)
    tekst = re.sub(r"\s+", " ", tekst)
    return tekst.strip()

def genereer_voorbeelden(categorie, voorbeelden):
    global huidige_api_sleutel_index
    prompt = (
        f"Generate a list of additional waste item examples in the '{categorie}' category. "
        f"Include only the names of the items, without any extra text or description. "
        f"Base the list on items like: {', '.join(voorbeelden)}."
    )

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        respons = model.generate_content(prompt)
        if respons and respons.text:
            items = [maak_item_tekst_schoon(item) for item in respons.text.split('\n') if item.strip()]
            return items
        else:
            print(f"Geen items gegenereerd voor: {categorie}")
            return []
    except Exception as e:
        if "429" in str(e):
            print(f"Limiy overschreden voor API-key {huidige_api_sleutel_index + 1}. volgende api...")
            huidige_api_sleutel_index = (huidige_api_sleutel_index + 1) % len(api_sleutels)
            configureer_api(api_sleutels[huidige_api_sleutel_index])
            return genereer_voorbeelden(categorie, voorbeelden)
        else:
            print(f"Fout bij het genereren van {categorie}: {e}")
            return []

gegenereerde_data = []

start_tijd = time.time()

while len(gegenereerde_data) < 10000:
    for categorie, voorbeelden in categorieën.items():
        print(f"Genereer items voor categorie: {categorie}")
        extra_voorbeelden = genereer_voorbeelden(categorie, voorbeelden)
        
        for item in extra_voorbeelden:
            if len(gegenereerde_data) >= 10000:
                break
            if item not in [invoer['Item'] for invoer in gegenereerde_data]:
                gegenereerde_data.append({"Item": item, "Categorie": categorie})
        
        print(f"Huidig totaal unieke items: {len(gegenereerde_data)}")

        if len(gegenereerde_data) >= 10000:
            break

    verstreken_tijd = time.time() - start_tijd
    print(f"Verstreken tijd: {verstreken_tijd:.2f} seconden.  totaal unieke items: {len(gegenereerde_data)}.\n")

df = pd.DataFrame(gegenereerde_data, columns=["Item", "Categorie"])

OUTPUT_bestandsnaam = "afval_items_gemini_Mark4.xlsx"
df.to_excel(OUTPUT_bestandsnaam, index=False)

totale_tijd = time.time() - start_tijd
print(f"Gegenereerde afvalitems zijn opgeslagen in {OUTPUT_bestandsnaam} met {len(df)} rijen.")
print(f"Totale tijd genomen: {totale_tijd:.2f} seconden.")
