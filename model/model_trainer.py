import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pylab
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Resizing

# Enabling memory growth to avoid CUDNN_STATUS_NOT_INITIALIZED on Windows
try:
    tf_gpus = tf.config.list_physical_devices("GPU")
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Set seed for experiment reproducibility
__SEED = 42
tf.random.set_seed(__SEED)
np.random.seed(__SEED)

__MINI_SPEECH_COMMANDS = "mini_speech_commands"
__DATA_PATH = "data/" + __MINI_SPEECH_COMMANDS
__DATA_PATH_EXTRACTED = __DATA_PATH + "_extracted"
__AUTOTUNE = tf.data.AUTOTUNE
__SAMPLES = 16000  # 1 sec at 16Khz
__BATCH_SIZE = 64
__EPOCHS = 15


def __build_data_dir():
    data_dir = pathlib.Path(__DATA_PATH_EXTRACTED + "/" + __MINI_SPEECH_COMMANDS)

    if not data_dir.exists():
        tf.keras.utils.get_file(
            "mini_speech_commands.zip",
            origin="http://storage.googleapis.com/download.tensorflow.org/data/" + __MINI_SPEECH_COMMANDS + ".zip",
            extract=True,
            cache_dir=".",
            cache_subdir="data")

        tf.io.gfile.remove("%s.zip" % __DATA_PATH)
        tf.io.gfile.remove("%s/%s/README.md" % (__DATA_PATH_EXTRACTED, __MINI_SPEECH_COMMANDS))
        tf.io.gfile.rmtree("%s/%s/go" % (__DATA_PATH_EXTRACTED, __MINI_SPEECH_COMMANDS))
        tf.io.gfile.rmtree("%s/%s/no" % (__DATA_PATH_EXTRACTED, __MINI_SPEECH_COMMANDS))
        tf.io.gfile.rmtree("%s/%s/yes" % (__DATA_PATH_EXTRACTED, __MINI_SPEECH_COMMANDS))
        tf.io.gfile.rmtree("%s/%s/" % (__DATA_PATH_EXTRACTED, "__MACOSX"))

    return pathlib.Path(__DATA_PATH_EXTRACTED + "/" + __MINI_SPEECH_COMMANDS), np.array(
        tf.io.gfile.listdir(str(data_dir)))


def __setup_files():
    filenames = tf.io.gfile.glob(str(__data_dir) + "/*/*")

    filenames = tf.random.shuffle(filenames)

    num_samples = len(filenames)

    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = int(0.1 * num_samples)

    return filenames[:train_size], filenames[train_size: train_size + val_size], filenames[-test_size:]


def __decode_audio(audio_binary):
    decoded_audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(decoded_audio, axis=-1)


def __get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You"ll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def __get_waveform_and_label(file_path):
    file_name = __get_label(file_path)

    audio_binary = tf.io.read_file(file_path)

    decoded_waveform = __decode_audio(audio_binary)

    return decoded_waveform, file_name


def __get_spectrogram(waveform):
    # Padding for files with less than __SAMPLES samples
    zero_padding = tf.zeros([__SAMPLES] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)

    equal_length = tf.concat([waveform, zero_padding], 0)

    spectrogram = tf.signal.stft(
        equal_length,
        frame_length=255,
        frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def __plot_waveforms():
    rows = 3
    cols = 3

    n = rows * cols

    fig, axes = plt.subplots(rows, cols, num="Waveform samples")
    fig.tight_layout()

    wm = plt.get_current_fig_manager()
    wm.window.state("zoomed")

    for i, (audio, label) in enumerate(__waveform_ds.take(n)):
        r = i // cols
        c = i % cols

        ax = axes[r][c]
        ax.plot(audio.numpy())

        label = label.numpy().decode("utf-8")

        ax.set_title(label)
        ax.set_ylabel("amplitude")
        ax.set_xlabel("sample")

    plt.show()


def __plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)

    height = log_spec.shape[0]

    width = log_spec.shape[1]

    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)

    Y = range(height)

    return ax.pcolormesh(X, Y, log_spec)


def __plot_spectrograms(spectrogram_ds):
    rows = 3
    cols = 3

    n = rows * cols

    fig, axes = plt.subplots(rows, cols, num="Spectrogram samples")
    fig.tight_layout()

    wm = plt.get_current_fig_manager()
    wm.window.state("zoomed")

    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        mesh = __plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title(__commands[label_id.numpy()])
        ax.set_ylabel("f(Hz)")
        ax.set_xlabel("sample")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("PSD (dB)")

    plt.show()


def __plot_evolution_of_metrics(history):
    metrics = history.history

    # loss and val_loss differ because the former is applied to the train set, and the latter the validation set.
    plt.plot(__history.epoch, metrics["loss"], metrics["val_loss"])
    plt.plot(__history.epoch, metrics["accuracy"], metrics["val_accuracy"])

    plt.title("Evolution of metrics")

    plt.legend(["Train loss", "Test loss", "Train accuracy", "Test accuracy"])
    plt.ylabel("Value")
    plt.xlabel("Epoch")

    fig = pylab.gcf()
    fig.canvas.manager.set_window_title("Evolution of metrics")

    plt.show()


def __plot_confusion_matrix(confusion_mtx):
    plt.figure(figsize=(10, 8))

    sns.heatmap(confusion_mtx,
                xticklabels=__commands,
                yticklabels=__commands,
                annot=True,
                fmt="g")

    plt.title("Confusion matrix")

    plt.xlabel("Prediction")
    plt.ylabel("Label")

    fig = pylab.gcf()
    fig.canvas.manager.set_window_title("Confusion matrix")

    plt.show()


def __get_spectrogram_and_label_id(audio, label):
    spectrogram = __get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == __commands)
    return spectrogram, label_id


def __preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(__get_waveform_and_label, num_parallel_calls=__AUTOTUNE)
    output_ds = output_ds.map(__get_spectrogram_and_label_id, num_parallel_calls=__AUTOTUNE)
    return output_ds


__data_dir, __commands = __build_data_dir()

print("Available commands:", __commands)

__num_labels = len(__commands)

__train_files, __val_files, __test_files = __setup_files()

print("Training set size", len(__train_files))
print("Validation set size", len(__val_files))
print("Test set size", len(__test_files))

__files_ds = tf.data.Dataset.from_tensor_slices(__train_files)
__waveform_ds = __files_ds.map(__get_waveform_and_label, num_parallel_calls=__AUTOTUNE)

__plot_waveforms()

__spectrogram_ds = __waveform_ds.map(__get_spectrogram_and_label_id, num_parallel_calls=__AUTOTUNE)

__plot_spectrograms(__spectrogram_ds)

__train_ds = __spectrogram_ds
__val_ds = __preprocess_dataset(__val_files)
__test_ds = __preprocess_dataset(__test_files)

__train_ds = __train_ds.batch(__BATCH_SIZE)
__val_ds = __val_ds.batch(__BATCH_SIZE)

__train_ds = __train_ds.cache().prefetch(__AUTOTUNE)
__val_ds = __val_ds.cache().prefetch(__AUTOTUNE)

for spectrogram, _ in __spectrogram_ds.take(1):
    __input_shape = spectrogram.shape

print("Input shape:", __input_shape)

__norm_layer = Normalization()
__norm_layer.adapt(__spectrogram_ds.map(lambda x, _: x))

__model = models.Sequential([
    layers.Input(shape=__input_shape),
    Resizing(32, 32),
    __norm_layer,
    layers.Conv2D(32, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),  # Prevents overfitting
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),  # Prevents overfitting
    layers.Dense(__num_labels),
])

__model.summary()

__model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

__history = __model.fit(
    __train_ds,
    validation_data=__val_ds,
    epochs=__EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    # Stop training when "val_loss" metric has stopped improving after 2 epochs.
)

__plot_evolution_of_metrics(__history)

__test_audio = []
__test_labels = []

for audio, label in __test_ds:
    __test_audio.append(audio.numpy())
    __test_labels.append(label.numpy())

__test_audio = np.array(__test_audio)
__test_labels = np.array(__test_labels)

__y_pred = np.argmax(__model.predict(__test_audio), axis=1)
__y_true = __test_labels

__test_acc = sum(__y_pred == __y_true) / len(__y_true)

print(f"Test set accuracy: {__test_acc:.0%}")

__confusion_mtx = tf.math.confusion_matrix(__y_true, __y_pred)

__plot_confusion_matrix(tf.math.confusion_matrix(__y_true, __y_pred))

__model_output_path = "model_output/"

if not os.path.exists(__model_output_path):
    os.makedirs(__model_output_path)

__model.save(__model_output_path + "model.keras")

__sample_file = __data_dir / "right/1aeef15e_nohash_1.wav"

__sample_ds = __preprocess_dataset([str(__sample_file)])

for spectrogram, label in __sample_ds.batch(1):
    prediction = __model(spectrogram)

    # Softmax function calculates the probabilities distribution of the event over ‘n’ different events.
    # In general way of saying, this function will calculate the probabilities of each target class over all
    # possible target classes. Later the calculated probabilities will be helpful for determining the target
    # class for the given inputs.
    plt.bar(__commands, tf.nn.softmax(prediction[0]))

    plt.title(f'Predictions for "{__commands[label[0]]}"')

    fig = pylab.gcf()
    fig.canvas.manager.set_window_title(f'Predictions for "{__commands[label[0]]}"')

    plt.show()

    print(tf.nn.softmax(prediction[0]))
