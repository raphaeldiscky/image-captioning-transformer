from cmath import log
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from dataset import (
    make_dataset,
    custom_standardization,
    reduce_dataset_dim,
    valid_test_split,
)
from custom_schedule import custom_schedule
from model import (
    get_cnn_model,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    ImageCaptioningModel,
)
from utility import save_tokenizer
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
from plot_training import plot_training
from settings import (
    BATCH_SIZE,
    DATE_NOW,
    EMBED_DIM,
    EPOCHS,
    FF_DIM,
    IMAGE_SIZE,
    NUM_HEADS,
    SAVE_DIR,
    VALID_SET_AUG,
    train_data_json_path,
    valid_data_json_path,
    text_data_json_path,
    REDUCE_DATASET,
    MAX_VOCAB_SIZE,
    SEQ_LENGTH,
    TRAIN_SET_AUG,
    TEST_SET,
)


# Load dataset
with open(train_data_json_path) as json_file:
    train_data = json.load(json_file)
with open(valid_data_json_path) as json_file:
    valid_data = json.load(json_file)
with open(text_data_json_path) as json_file:
    text_data = json.load(json_file)

# For reduce number of images in the dataset
if REDUCE_DATASET:
    train_data, valid_data = reduce_dataset_dim(train_data, valid_data)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

# Define tokenizer of Text Dataset
tokenizer = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

# Adapt tokenizer to Text Dataset
tokenizer.adapt(text_data)

# Define vocabulary size of Dataset
VOCAB_SIZE = 27277
print(VOCAB_SIZE)

# 20k images for validation set and 13432 images for test set
valid_data, test_data = valid_test_split(valid_data)

print("Number of validation samples after splitting with test set: ", len(valid_data))
print("Number of test samples: ", len(test_data))

# Setting batch dataset
train_dataset = make_dataset(
    list(train_data.keys()),
    list(train_data.values()),
    data_aug=TRAIN_SET_AUG,
    tokenizer=tokenizer,
)

print("TRAIIIN", train_dataset)

valid_dataset = make_dataset(
    list(valid_data.keys()),
    list(valid_data.values()),
    data_aug=VALID_SET_AUG,
    tokenizer=tokenizer,
)
if TEST_SET:
    test_dataset = make_dataset(
        list(test_data.keys()),
        list(test_data.values()),
        data_aug=False,
        tokenizer=tokenizer,
    )


# Define Model
cnn_model = get_cnn_model()

encoder = TransformerEncoderBlock(
    embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
)
decoder = TransformerDecoderBlock(
    embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Create a learning rate schedule
lr_scheduler = custom_schedule(EMBED_DIM)
optimizer = keras.optimizers.Adam(
    learning_rate=lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

# Compile the model
caption_model.compile(optimizer=optimizer, loss=cross_entropy)

# Fit the model
history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)

plot_training(history)

# Compute definitive metrics on train/valid set
train_metrics = caption_model.evaluate(train_dataset, batch_size=BATCH_SIZE)
valid_metrics = caption_model.evaluate(valid_dataset, batch_size=BATCH_SIZE)
if TEST_SET:
    test_metrics = caption_model.evaluate(test_dataset, batch_size=BATCH_SIZE)

print(
    "Train Loss = %.4f - Train Accuracy = %.4f" % (train_metrics[0], train_metrics[1])
)
print(
    "Valid Loss = %.4f - Valid Accuracy = %.4f" % (valid_metrics[0], valid_metrics[1])
)
if TEST_SET:
    print(
        "Test Loss = %.4f - Test Accuracy = %.4f" % (test_metrics[0], test_metrics[1])
    )

# Create new directory for saving model
NEW_DIR = SAVE_DIR + DATE_NOW
os.mkdir(NEW_DIR)

# Save training history under the form of a json file
history_dict = history.history
json.dump(history_dict, open(SAVE_DIR + "{}/history.json".format(DATE_NOW), "w"))

# Save weights model
caption_model.save_weights(SAVE_DIR + "{}/model_weights_coco.h5".format(DATE_NOW))

# Save config model train
config_train = {
    "IMAGE_SIZE": IMAGE_SIZE,
    "MAX_VOCAB_SIZE": MAX_VOCAB_SIZE,
    "SEQ_LENGTH": SEQ_LENGTH,
    "EMBED_DIM": EMBED_DIM,
    "NUM_HEADS": NUM_HEADS,
    "FF_DIM": FF_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "VOCAB_SIZE": VOCAB_SIZE,
}

json.dump(config_train, open(SAVE_DIR + "{}/config_train.json".format(DATE_NOW), "w"))

# Save Tokenizer model
save_tokenizer(tokenizer, NEW_DIR)
