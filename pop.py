import numpy as np
from gtts import gTTS
from scipy.io import wavfile
from pathlib import Path
import os

class LinearInterpolator:
    def __call__(self, values, index):
        low = int(index)
        high = int(np.ceil(index))
        if low == high:
            return values[low]
        return (index - low) * values[high % values.shape[0]] + (high - index) * values[low]

class WavetableOscillator:
    def __init__(self, wavetable, sampling_rate, interpolator):
        self.wavetable = wavetable
        self.sampling_rate = sampling_rate
        self.interpolator = interpolator
        self.wavetable_index = 0.0
        self.__frequency = 0

    def fill(self, audio_block, from_index=0, to_index=-1):
        for i in range(from_index, to_index % audio_block.shape[0]):
            audio_block[i] = self.get_sample()
        return audio_block

    def get_sample(self):
        sample = self.interpolator(self.wavetable, self.wavetable_index)
        self.wavetable_index = (self.wavetable_index + self.wavetable_increment) % self.wavetable.shape[0]
        return sample

    @property
    def frequency(self):
        return self.__frequency

    @frequency.setter
    def frequency(self, value):
        self.__frequency = value
        self.wavetable_increment = self.wavetable.shape[0] * self.frequency / self.sampling_rate
        if self.frequency <= 0:
            self.wavetable_index = 0.0

class Voice:
    def __init__(self, sampling_rate, gain=-10):
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.oscillators = []

    def synthesize(self, frequency, duration_seconds):
        buffer = np.zeros((duration_seconds * self.sampling_rate,))
        if np.isscalar(frequency):
            frequency = np.ones_like(buffer) * frequency

        for i in range(len(buffer)):
            for oscillator in self.oscillators:
                oscillator.frequency = frequency[i]
                buffer[i] += oscillator.get_sample()
        amplitude = 10 ** (self.gain / 20)
        buffer *= amplitude
        buffer = fade_in_out(buffer)
        return buffer

def fade_in_out(signal, fade_length=1000):
    fade_in_envelope = (1 - np.cos(np.linspace(0, np.pi, fade_length))) * 0.5
    fade_out_envelope = np.flip(fade_in_envelope)

    if signal.ndim == 2:
        fade_in_envelope = fade_in_envelope[:, np.newaxis]
        fade_out_envelope = fade_out_envelope[:, np.newaxis]

    signal[:fade_length, ...] = np.multiply(signal[:fade_length, ...], fade_in_envelope)
    signal[-fade_length:, ...] = np.multiply(signal[-fade_length:, ...], fade_out_envelope)

    return signal

def generate_wavetable(length, f):
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(2 * np.pi * i / length)
    return wavetable

def output_wavs(signal, name, sampling_rate, table):
    output_dir = Path('wavetable-synthesis-python')
    output_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(
        output_dir / f'{name}_table.wav',
        sampling_rate,
        table.astype(np.float32))
    wavfile.write(
        output_dir / f'{name}.wav',
        sampling_rate,
        signal.astype(np.float32))

def adjust_pitch(buffer, pitch_factor=1.3):  # Increased pitch factor for higher sound
    indices = np.round(np.arange(0, len(buffer), pitch_factor)).astype(int)
    # Ensure indices stay within the bounds of the buffer
    indices = np.clip(indices, 0, len(buffer) - 1)
    return buffer[indices]

def adjust_speed(buffer, speed_factor=1.3):  # Increased speed factor for faster speech
    indices = np.round(np.arange(0, len(buffer), speed_factor)).astype(int)
    # Ensure indices stay within the bounds of the buffer
    indices = np.clip(indices, 0, len(buffer) - 1)
    return buffer[indices]

# ฟังก์ชันสำหรับสร้างเสียงจากข้อความ
def synthesize_text_to_speech(text, filename='output_tts.wav'):
    # ใช้ gTTS เพื่อแปลงข้อความเป็นเสียง
    tts = gTTS(text=text, lang='th', slow=False)  # ใช้ slow=False สำหรับการพูดเร็ว
    tts.save('temp.mp3')  # บันทึกไฟล์ชั่วคราวเป็นไฟล์ MP3

    # แปลงไฟล์ MP3 เป็น WAV
    os.system(f'ffmpeg -i temp.mp3 -ar 22050 {filename}')  # ใช้ ffmpeg แปลงไฟล์
    os.remove('temp.mp3')  # ลบไฟล์ชั่วคราวหลังจากแปลง

    print(f"Saved synthesized speech to {filename}")

def main():
    sampling_rate = 44100
    wavetable_size = 64
    synth = Voice(sampling_rate, gain=-5)  # Increased gain for brightness

    # Generate sine wave
    sine_table = generate_wavetable(wavetable_size, np.sin)
    synth.oscillators += [WavetableOscillator(sine_table, sampling_rate, LinearInterpolator())]
    sine = synth.synthesize(frequency=880, duration_seconds=3)

    # Adjust pitch and speed for a joyful tone
    sine = adjust_pitch(sine, pitch_factor=2.0)  # Increased pitch factor for higher sound
    sine = adjust_speed(sine, speed_factor=2.7)  # Increased speed factor for faster speech

    output_wavs(sine, 'sine', sampling_rate, sine_table)

    # สร้างเสียงที่มีอารมณ์
    text = "สวัสดี"
    synthesize_text_to_speech(text, filename='sawasdee_joyful.wav')

if __name__ == "__main__":
    main()

