<p align="center">
  <img src="/assets/screenshot.png">
</p>

# PY-SNAKE

A voice-commanded snake game featuring a Convolutional Neural Network model built on top of TensorFlow framework.

Supported voice-commands are:

- **Up**
- **Down**
- **Left**
- **Right**
- **Stop** -- stops and resets the game.

# Architecture

This project is divided into two packages: **model** and **game**. 

## Model package

Contains [model_trainer.py](model/model_trainer.py) that will train and produce a CNN model into 
**/model/model_output** folder when executed. Provides a step-by-step training, showing how audio samples
become waveforms then spectrograms to be fed to a CNN model. Also presents the statistic scores for
the trained model.

## Game package

Is composed of:

 - [snake_game.py](game/snake_game.py) - provides game screen drawing.
 - [voice_detector.py](game/voice_detector.py) - detects and record voice segments from microphone.
 - [command_processor.py](game/command_processor.py) - translate voice commands into game commands using the 
pre-trained CNN model.

# How to install required libs:

PyAudio doesn't provide a straight forward installation through pip, so:

### For 64 bits windows users:

    pip install .\libs\PyAudio-0.2.11-cp38-cp38-win_amd64.whl

### For linux users:

    sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
    sudo apt-get install ffmpeg libav-tools
    sudo pip install pyaudio

### For 32 bits windows users:
    
    pip install .\libs\PyAudio-0.2.11-cp38-cp38-win32.whl


All other libs installations should run fine from [requirements.txt](requirements.txt) file. 


# How to compile and run:

A pre-trained CNN model is already in place, but if you want to fine tune and build your own CNN model, check [model_trainer.py](model/model_trainer.py) class, 
apply your changes and run it. It will generate a custom CNN model in **model/model_output** folder. This exported
CNN model will be used by the game.

The main class of the game is [main.py](__main__.py), run it, wait for thing to initialize and shout a direction command.

