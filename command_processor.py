import threading

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE


class CommandProcessor(threading.Thread):

    def __init__(self, game, voice_detector, model):
        threading.Thread.__init__(self)

        self.__game = game
        self.__voice_detector = voice_detector
        self.__model = model

        self.__command_dict = {
            0: self.__game.go_down,
            1: self.__game.do_nothing,
            2: self.__game.go_left,
            3: self.__game.do_nothing,
            4: self.__game.go_right,
            5: self.__game.do_nothing,
            6: self.__game.go_up,
            7: self.__game.do_nothing
        }

    def __decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def __build_waveform(self, audio_binary):
        waveform = self.__decode_audio(audio_binary)
        return waveform

    def __build_spectrogram(self, waveform):
        # Concatenate audio with padding so that all audio clips will be of the same length
        waveform = tf.cast(waveform, tf.float32)

        spectrogram = tf.signal.stft(
            waveform,
            frame_length=255,
            frame_step=128)

        spectrogram = tf.abs(spectrogram)

        spectrogram = tf.expand_dims(spectrogram, -1)

        return spectrogram

    def __build_spectrogram_dataset(self, audio_binary):

        audio_binary_ds = tf.data.Dataset.from_tensor_slices([audio_binary])

        spectrogram_ds = audio_binary_ds.map(self.__build_waveform, num_parallel_calls=AUTOTUNE)

        return spectrogram_ds.map(self.__build_spectrogram, num_parallel_calls=AUTOTUNE)

    def run(self):

        while True:

            audio_binary = self.__voice_detector.record_to_file()

            spectrogram_ds = self.__build_spectrogram_dataset(audio_binary)

            for spectrogram in spectrogram_ds.batch(1):
                prediction = self.__model(spectrogram)

                predictions = tf.nn.softmax(prediction[0])

                self.__command_dict[predictions.numpy().argmax()]()
