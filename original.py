import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from sklearn.utils import resample

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def create_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, dtype='float32'),  # Mixed precision requires float32 for final dense layers
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Dense(128, dtype='float32'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    return model

def prepare_balanced_data(metadata_path, image_dirs, img_size=(224, 224)):
    df = pd.read_csv(metadata_path)
    
    def get_image_path(row):
        image_id = str(row['image_id'])
        for dir_path in image_dirs:
            full_path = os.path.join(dir_path, f"{image_id}.jpg")
            if os.path.exists(full_path):
                return full_path
        return None
    
    df['file_path'] = df.apply(get_image_path, axis=1)
    df = df.dropna(subset=['file_path'])
    
    # Balance the dataset
    df_list = []
    min_samples = min(df['dx'].value_counts())
    
    for class_name in df['dx'].unique():
        df_class = df[df['dx'] == class_name]
        if len(df_class) < 1000:  # For minority classes
            n_samples = min(1000, len(df_class) * 3)  # Oversample but cap at 1000
        else:  # For majority classes
            n_samples = min(len(df_class), 1000)  # Cap at 1000
        
        df_balanced = resample(df_class, 
                             replace=True if len(df_class) < n_samples else False,
                             n_samples=n_samples,
                             random_state=42)
        df_list.append(df_balanced)
    
    balanced_df = pd.concat(df_list)
    
    print("\nBalanced class distribution:")
    print(balanced_df['dx'].value_counts())
    
    # Split into train/validation/test
    train_df = balanced_df.sample(frac=0.8, random_state=42)
    remaining_df = balanced_df.drop(train_df.index)
    val_df = remaining_df.sample(frac=0.5, random_state=42)
    test_df = remaining_df.drop(val_df.index)
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="file_path",
        y_col="dx",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=img_size
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="file_path",
        y_col="dx",
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=img_size
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="file_path",
        y_col="dx",
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=img_size
    )
    
    return train_generator, validation_generator, test_generator

def train_model(metadata_path, image_dirs, model_save_path='skin_lesion_model.keras'):
    train_generator, validation_generator, test_generator = prepare_balanced_data(metadata_path, image_dirs)
    
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes=num_classes)
    
    # Initial training
    initial_learning_rate = 0.001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\nPhase 1: Initial training...")
    history1 = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nPhase 2: Fine-tuning...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze earlier layers
    for layer in base_model.layers[:-30]:  # Train only the last 30 layers
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history[key]
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    
    return model, history, test_acc

# Main execution
base_dir = "/kaggle/input/data-set"
metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")
image_dirs = [
    os.path.join(base_dir, "HAM10000_images_part_1"),
    os.path.join(base_dir, "HAM10000_images_part_2")
]
model_save_path = "/kaggle/working/skin_lesion_model.keras"

model, history, test_acc = train_model(metadata_path, image_dirs, model_save_path)

# Print metrics
best_epoch = np.argmax(history['val_accuracy'])
print("\nFinal Metrics:")
print(f"Best Training Accuracy: {history['accuracy'][best_epoch]*100:.2f}%")
print(f"Best Validation Accuracy: {max(history['val_accuracy'])*100:.2f}%")
print(f"Final Training Accuracy: {history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {history['val_accuracy'][-1]*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.show()
