import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, embed_dim)
        position_embedding_matrix = self.get_position_encoding(
            sequence_length, embed_dim
        )
        self.word_embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[word_embedding_matrix],
            trainable=False,
        )
        self.position_embedding_layer = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim,
            weights=[position_embedding_matrix],
            trainable=False,
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for pos in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[pos, 2 * i] = np.sin(pos / denominator)
                P[pos, 2 * i + 1] = np.cos(pos / denominator)
        return P

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
