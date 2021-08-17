import tensorflow as tf

from game.command_processor import CommandProcessor
from game.snake_game import SnakeGame
from game.voice_detector import VoiceDetector

# Enabling memory growth to avoid CUDNN_STATUS_NOT_INITIALIZED on Windows
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


def load_model():
    model = tf.keras.models.load_model('model')
    model.summary()
    return model


def main():
    game = SnakeGame()

    voice_detector = VoiceDetector()

    model = load_model()

    command_processor = CommandProcessor(game, voice_detector, model)

    command_processor.start()

    game.run()


if __name__ == "__main__":
    main()
