import io
import time
import wave
from array import array
from struct import pack
from sys import byteorder

import pyaudio


class VoiceDetector:

    __SILENCE_THRESHOLD = 1500

    __CHUNK_SIZE = 1024

    __FORMAT = pyaudio.paInt16

    __RATE = 16000

    __MAX_VOLUME = 16384

    __SILENCE_TIME = 0.25

    # Returns "True" if below the "silent" threshold
    def __is_silent(self, snd_data):
        return max(snd_data) < self.__SILENCE_THRESHOLD

    # Average the volume out
    def __normalize(self, snd_data):
        times = float(self.__MAX_VOLUME) / max(abs(i) for i in snd_data)

        r = array("h")

        for i in snd_data:
            r.append(int(i * times))

        return r

    def __trim(self, snd_data):
        snd_started = False
        r = array("h")

        for i in snd_data:
            if not snd_started and abs(i) > self.__SILENCE_THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim the blank spots at the start and end
    def __trim_silence(self, snd_data):

        # Trim to the left
        snd_data = self.__trim(snd_data)

        # Trim to the right
        snd_data.reverse()

        snd_data = self.__trim(snd_data)

        snd_data.reverse()

        return snd_data

    # Add silence to the start and end of "snd_data" of length "seconds" (float)
    def __add_silence(self, snd_data, seconds):
        silence = [0] * int(seconds * self.__RATE)

        r = array("h", silence)

        r.extend(snd_data)

        r.extend(silence)

        return r

    # 2-bytes little-endian short
    def __pack_data(self, data):
        return pack("<" + ("h" * len(data)), *data)

    # Record a word or words from the microphone and
    # return the data as an array of signed shorts.
    # Normalizes the audio, trims silence from the
    # start and end, and pads with 0.5 seconds of
    # blank sound to make sure VLC et al can play
    # it without getting chopped off.
    def record(self):

        p = pyaudio.PyAudio()

        stream = p.open(format=self.__FORMAT,
                        channels=1,
                        rate=self.__RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=self.__CHUNK_SIZE)

        num_silent = 0

        snd_started = False

        r = array("h")

        while True:
            # little endian, signed short
            snd_data = array("h", stream.read(self.__CHUNK_SIZE))

            if byteorder == "big":
                snd_data.byteswap()

            r.extend(snd_data)

            silent = self.__is_silent(snd_data)

            if silent and snd_started:
                num_silent += 1

            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > 1:
                break

        sample_width = p.get_sample_size(self.__FORMAT)

        stream.stop_stream()
        stream.close()
        p.terminate()

        r = self.__normalize(r)

        r = self.__trim_silence(r)

        r = self.__add_silence(r, self.__SILENCE_TIME)

        return sample_width, r

    # Records from the microphone and outputs the resulting data to "path"
    def record_to_file(self):

        print("Say")

        file_in_memory = io.BytesIO()

        t1 = round(time.time() * 1000)
        sample_width, data = self.record()
        t2 = round(time.time() * 1000)
        print("1: ", t2 - t1)

        # 2-bytes little-endian short
        data = self.__pack_data(data)

        wf = wave.open(file_in_memory, "wb")

        wf.setnchannels(1)

        wf.setsampwidth(sample_width)

        wf.setframerate(self.__RATE)

        wf.writeframes(data)

        wf.close()

        file_in_memory.seek(0)

        return file_in_memory.read()
