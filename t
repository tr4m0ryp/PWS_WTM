# material_recognition_model_with_label_smoothing.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Stap 1: Data Laden en Preprocessen

# Lees de CSV-bestanden
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

# Controleer of de kolommen correct zijn
print("Kolommen in train_df:", train_df.columns)

# Labels Encoderen
label_encoder = LabelEncoder()

# Combineer labels uit trainings- en validatiesets voor consistente encoding
all_labels = pd.concat([train_df['material'], val_df['material']], axis=0)

label_encoder.fit(all_labels)

# Transformeer labels
train_df['label'] = label_encoder.transform(train_df['material'])
val_df['label'] = label_encoder.transform(val_df['material'])
test_df['label'] = label_encoder.transform(test_df['material'])

# One-hot encoding van labels voor label smoothing
num_classes = len(label_encoder.classes_)

train_df['label_ohe'] = list(tf.keras.utils.to_categorical(train_df['label'], num_classes))
val_df['label_ohe'] = list(tf.keras.utils.to_categorical(val_df['label'], num_classes))
test_df['label_ohe'] = list(tf.keras.utils.to_categorical(test_df['label'], num_classes))

# Bewaar de mapping van labels voor later gebruik
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Functie om afbeeldingen te preprocessen
def preprocess_image(image_path):
    # Lees het beeldbestand
    image = tf.io.read_file(image_path)
    # Decodeer het beeld
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize het beeld
    image = tf.image.resize(image, [224, 224])
    # Data Augmentatie toepassen tijdens training
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Normaliseer de pixelwaarden
    image = image / 255.0
    return image

def preprocess_image_test(image_path):
    # Preprocess zonder augmentatie voor validatie en test
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image

# Stap 2: Creëer TensorFlow Datasets met tf.data

def create_dataset(df, batch_size=32, shuffle=True, augment=True):
    image_paths = df['image_path'].values
    labels = np.array(df['label_ohe'].tolist())
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if augment:
        dataset = dataset.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x, y: (preprocess_image_test(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Maak de datasets
batch_size = 32

train_dataset = create_dataset(train_df, batch_size=batch_size, shuffle=True, augment=True)
val_dataset = create_dataset(val_df, batch_size=batch_size, shuffle=False, augment=False)
test_dataset = create_dataset(test_df, batch_size=batch_size, shuffle=False, augment=False)

# Stap 3: Model Architectuur Definiëren

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Bevries de basislaag
base_model.trainable = False

# Voeg aangepaste lagen toe
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Verhoogde Dropout voor regularisatie
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Model Overzicht
model.summary()

# Stap 4: Model Compileren

# Gebruik van Label Smoothing in de loss functie
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=loss_fn,
    metrics=['accuracy']
)

# Stap 5: Model Trainen

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

epochs = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Stap 6: Model Evalueren

# Evalueer op de testset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Nauwkeurigheid: {test_accuracy * 100:.2f}%")

# Confusion Matrix en Classification Report

# Voorspellingen genereren
y_pred = []
y_true = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels, axis=1))

# Converteer labels terug naar labelnamen
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# Classification Report
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Echte labels')
plt.xlabel('Voorspelde labels')
plt.title('Confusion Matrix')
plt.show()

# Stap 7: Model Fine-tunen

# Ontdooien van lagen voor fine-tuning
base_model.trainable = True

# Stel in tot welke laag je het model wilt ontdooien
fine_tune_at = 100  # Pas dit aan op basis van het model

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Hercompileren van het model met een lagere leersnelheid
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

# Vervolg Training
fine_tune_epochs = 5
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Stap 8: Model Opslaan

model.save('materiaalherkenningsmodel_final.h5')
print("Model is succesvol opgeslagen als 'materiaalherkenningsmodel_final.h5'.")

# Stap 9: Monitoring van Confidentie Scores

# Identificeer voorbeelden met lage confidentie
threshold = 0.5  # Stel een drempel in voor lage confidentie
low_confidence_indices = []

for idx, (images, labels) in enumerate(train_dataset):
    preds = model.predict(images)
    confidences = np.max(preds, axis=1)
    for i, confidence in enumerate(confidences):
        if confidence < threshold:
            image_index = idx * batch_size + i
            low_confidence_indices.append(image_index)

print(f"Aantal trainingsvoorbeelden met lage confidentie (<{threshold}): {len(low_confidence_indices)}")

# (Optioneel) Inspecteer deze voorbeelden handmatig om mislabeling te identificeren

# Stap 10: Aanbevelingen voor Verdere Verbetering

# Overweeg om de geïdentificeerde voorbeelden te herzien en de labels te corrigeren indien nodig.
# Dit kan de modelprestaties verder verbeteren.

