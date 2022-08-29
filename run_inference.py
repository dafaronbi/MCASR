from distutils.filelist import FileList
import glob
import argparse
from sunau import AUDIO_FILE_ENCODING_ADPCM_G722
import tensorflow_io as tfio
import tensorflow as tf
from tensorflow.keras import layers, losses
import librosa
import os
from matplotlib import pyplot as plt
import model
import numpy as np
from losses import SpectralLoss
import scipy

SAMPLING_RATE = 16000

#make arg parser for files to load
parser = argparse.ArgumentParser(description="file used to train inference")
parser.add_argument('--folder', dest='folder', help='file where data is stored')
args = parser.parse_args()

# SAMPLING_RATE = 16000

# audio,_ = librosa.load(args.file)

# audio = audio[:80000]

# #pad audio to be at least 5 seconds
# zero_padding = np.zeros(80000 - audio.shape[0], dtype=np.float32)

# audio = np.append(zero_padding, audio)

# data = tf.data.Dataset.from_tensor_slices(audio)
# data = data.map(set_shape)

def input_pipeline():

    #load in data from
    wav_data = tf.data.Dataset.list_files(args.folder + "/*.wav")
    mp3_data = tf.data.Dataset.list_files(args.folder + "/*.mp3")

    read_audio = lambda x: tf.py_function(load_audio,[x], tf.float64)
    data = wav_data.map(read_audio)
    data = data.map(set_shape)

    data = data.cache()
    # data = data.shuffle(buffer_size=1000)
    data = data.batch(16)
    data = data.prefetch(8) 

    return data

def load_audio(filepath):

    y,_ = librosa.load(filepath.numpy(), sr=SAMPLING_RATE)

    y = y[:80000]

    #pad audio to be at least 5 seconds
    zero_padding = np.zeros(80000 - y.shape[0], dtype=np.float32)

    y = np.append(zero_padding, y)
    
    return y

def set_shape(audio):
    audio.set_shape((80000))
    return {'audio_input':audio, 'f0_input': 440}, {'audio_output':audio, 'f0_output': 440}

def main():
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    data = input_pipeline()
    m = model.Autoencoder(64,SAMPLING_RATE)
    # m.build()
    m.compile(optimizer='adam', 
            loss={'audio_output':SpectralLoss(),'f0_output':tf.keras.losses.MeanSquaredError()},
            loss_weights = {'audio_output':0.8,'f0_output':0.2})
    m.load_weights(".")
    out = m.predict(data)
    scipy.io.wavfile.write("output_file", SAMPLING_RATE, out['audio_output'][0])

if __name__ == "__main__":
    main()