from ast import Pass
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import tensorflow as tf
import librosa

latent_dim = 64 

class f0_extract_layer(tf.keras.layers.Layer):
  def __init__(self,sr):
    super(f0_extract_layer, self).__init__()
    self.sample_rate = sr

  def build(self, input_shape):
    pass

  def call(self, inputs):
    f0s = librosa.yin(inputs.numpy(),fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),sr=self.sample_rate)
    mean_f0 = tf.reduce_mean(tf.convert_to_tensor(f0s), axis=1, keepdims=True)
    return mean_f0

def encoder(latent_dim):
  #audio encoder
  audio_inp = layers.Input((80000))
  audio = layers.Reshape([80000,1])(audio_inp)
  audio = layers.GRU(512)(audio)
  audio = layers.Dense(1024, activation='relu')(audio)
  audio = layers.Dense(1024, activation='relu')(audio)
  audio = layers.Dense(1024, activation='relu')(audio)


  #f0 encoder
  f0_inp = layers.Input((1))
  f0 = layers.Dense(512, activation='relu')(f0_inp)
  f0 = layers.Dense(512, activation='relu')(f0)

  #combine the two
  comb = layers.Concatenate()([audio,f0])
  comb = layers.Dense(latent_dim, activation='relu')(f0)

  return Model(inputs={'audio_input':audio_inp, 'f0_input': f0_inp}, outputs=[comb])

class Autoencoder(Model):
  def __init__(self, latent_dim,sr):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    # self.encoder = tf.keras.Sequential([
    #   layers.Flatten(),
    #   layers.Dense(latent_dim, activation='relu', input_shape=(80000,1))
    # ])

    self.encoder = encoder(latent_dim)
    self.decoder = tf.keras.Sequential([
      layers.Dense(512, activation='sigmoid'),
      layers.Reshape([512,1]),
      layers.GRU(512),
      layers.Dense(1024, activation='sigmoid'),
      layers.Dense(80000, activation='sigmoid')
    ])
    self.f0_extract = tf.keras.Sequential([f0_extract_layer(sr)])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    f0 = self.f0_extract(decoded)
    return {'audio_output':decoded,'f0_output':f0}