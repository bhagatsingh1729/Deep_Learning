
import os
import tensorflow as tf

os.makedirs("checkpoints", exist_ok=True)

callbacks = [

    # Save the best performing model
    tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/best_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Save the latest model after every epoch
    tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/latest_model.keras",
        save_best_only=False,
        save_freq="epoch"
    ),

    # Backup training state for recovery
    tf.keras.callbacks.BackupAndRestore(
        backup_dir="checkpoints/training_backup",
        save_freq="epoch",
        delete_checkpoint=False
    ),

    # Stop when validation loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

import keras

# Recommended: Modern Keras 3 (.keras) format
model = keras.saving.load_model("mnist_ann.keras")
"""
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)"""

