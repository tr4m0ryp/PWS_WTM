import pandas as pd
import random

file_path = r'C:\Users\Moussa (CCNV)\Downloads\PWS\DATA_gathering\Wordlist_Generater\Generated_Item_List\waste_items_gemini_Mark6.xlsx'
df = pd.read_excel(file_path)

header = df.iloc[0]  
data = df.iloc[1:].sample(frac=1).reset_index(drop=True) 

shuffled_df = pd.concat([header.to_frame().T, data], ignore_index=True)

output_file_path = 'shuffled_WordList_Mark7.xlsx'
shuffled_df.to_excel(output_file_path, index=False)

print(f"Het nieuwe geshuffelde bestand is opgeslagen als {output_file_path}")
