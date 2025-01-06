import pandas as pd

# Pad naar je originele CSV-bestand
input_csv = 'path/to/your/dataset.csv'  # Vervang dit door het pad naar je CSV-bestand

# Lees het CSV-bestand in
df = pd.read_csv(input_csv)

# Shuffle de DataFrame
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Opslaan van het gehusselde DataFrame als nieuw CSV-bestand
df_shuffled.to_csv('shuffled_dataset.csv', index=False)
