import numpy as numpy
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

import voice_detector


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_waveform(audio_binary):
    waveform = decode_audio(audio_binary)
    return waveform


def get_spectrogram(waveform):
    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)

    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, -1)

    return spectrogram


# Enabling memory growth to avoid CUDNN_STATUS_NOT_INITIALIZED on Windows
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

model = tf.keras.models.load_model('model')
# model.summary()

while True:

    audio_binary = voice_detector.record_to_file()

    audio_binary_ds = tf.data.Dataset.from_tensor_slices([audio_binary])

    output_ds = audio_binary_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)

    output_ds = output_ds.map(get_spectrogram, num_parallel_calls=AUTOTUNE)

    for spectrogram in output_ds.batch(1):
        prediction = model(spectrogram)

        predictions = tf.nn.softmax(prediction[0])

        # print('Down', predictions[0])
        # print('Go', predictions[1])
        # print('Left', predictions[2])
        # print('No', predictions[3])
        # print('Right', predictions[4])
        # print('Stop', predictions[5])
        # print('Up', predictions[6])
        # print('Yes', predictions[7])

        result = numpy.argmax(predictions.numpy())

        print(result)

#


# voice_detector = threading.Thread(target=record(), args=(1,), daemon=True)
# voice_detector.start()
