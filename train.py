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

SAMPLING_RATE = 16000

#make arg parser for files to load
parser = argparse.ArgumentParser(description="parameters needed to train model")
parser.add_argument('--data_folder', dest='folder', help='folder where data is stored')
args = parser.parse_args()

def callbacks():

    ckpoint_save_model = tf.keras.callbacks.ModelCheckpoint(
    filepath=".",
    verbose=1,
    save_weights_only=True,
    save_freq=14*8,
    )

    ckpoint_tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    return [ckpoint_save_model, ckpoint_tensorboard]


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
    cbs = callbacks()
    m = model.Autoencoder(64,SAMPLING_RATE)
    m.summary()
    m.compile(optimizer='adam', 
            loss={'audio_output':SpectralLoss(),'f0_output':tf.keras.losses.MeanSquaredError()},
            loss_weights = {'audio_output':0.8,'f0_output':0.2})
    m.fit(data, epochs=500,shuffle=True,callbacks=cbs)

if __name__ == "__main__":
    main()