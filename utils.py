import tensorflow as tf
from settings_train import IMAGE_SIZE
from models import (
    get_cnn_model,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    ImageCaptioningModel,
)
from datasets import read_image_inf
import numpy as np
import json


def save_tokenizer(tokenizer, path_save):
    input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output = tokenizer(input)
    model = tf.keras.Model(input, output)
    model.save(path_save + "/tokenizer", save_format="tf")


def get_inference_model(model_config_path):
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]
    CNN_MODEL = model_config["CNN_MODEL"]

    cnn_model = get_cnn_model(CNN_MODEL)
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )

    # it's necessary for init model -> without it, weights subclass model fails
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, training, decoder_input])

    return caption_model


def generate_caption(image_path, caption_model, tokenizer, SEQ_LENGTH):
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    # read the image from the disk
    img = read_image_inf(image_path)

    # pass the image to the CNN
    img = caption_model.cnn_model(img)

    # pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # generate the caption using the Transformer decoder
    decoded_caption = "<start>"
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("<start> ", "")