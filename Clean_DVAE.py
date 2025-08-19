import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import Lambda, Layer, Input, Dense, GaussianNoise
from sklearn.metrics.pairwise import cosine_similarity

# Deep Variational Autoencoder (DVAE)
class VAE(Model):
    def __init__(self, original_dim, latent_dim=3, dropout_rate=0.2):
        super(VAE, self).__init__()
        self.encoder_layers = [ # Encoding Layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate)
        ]
        self.z_mean_layer = Dense(latent_dim) # Latent space
        self.z_log_var_layer = Dense(latent_dim)
        self.sampling = Sampling()
        self.decoder_layers = [ # Decoder Layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(original_dim)
        ]
    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        z = self.sampling([z_mean, z_log_var])
        return z, z_mean, z_log_var

    def decode(self, z):
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def call(self, inputs):
        z, _, _ = self.encode(inputs)
        return self.decode(z)
# Training step with VAE loss - reconstruction and KL Divergence
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encode(data)
            reconstruction = self.decode(z)
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}

    def get_encoder(self, input_dim):
        inputs = Input(shape=(input_dim,))
        h = inputs
        for layer in self.encoder_layers:
            h = layer(h)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        z = self.sampling([z_mean, z_log_var])
        return Model(inputs, z, name="encoder")