from tkinter import *
import pygame  
from tkinter import filedialog
import time
from mutagen.mp3 import MP3
import tkinter.ttk as ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wave
import os
import threading
from pydub import AudioSegment
from spleeter.separator import Separator
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
from tkinter import StringVar

root = Tk()
root.title('Sound Dynamiction Cross X')
root.geometry("800x800")
root.configure(bg="#f0f0f0")  


avg_amplitude_var = StringVar()
avg_frequency_var = StringVar()
avg_harmonics_var = StringVar()


pygame.mixer.init()


paused = False
stopped = True
song_length = 0
canvas = None  


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


def load_songs():
    songs = filedialog.askopenfilenames(title="Select Songs", filetypes=(("MP3 Files", "*.mp3"), ("WAV Files", "*.wav")))
    for song in songs:
        song_box.insert(END, song)

def convert_to_wav(file_path):
    if file_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(file_path)
        wav_path = os.path.splitext(file_path)[0] + ".wav"
        sound.export(wav_path, format="wav")
        return wav_path
    return file_path

def compute_harmonics(sound_wave, rate):
    fft_spectrum = np.abs(np.fft.fft(sound_wave))[:len(sound_wave) // 2]
    freq_axis = np.fft.fftfreq(len(sound_wave), 1 / rate)[:len(sound_wave) // 2]
    fundamental_freq = np.argmax(fft_spectrum)
    harmonics = [fundamental_freq * (i + 1) for i in range(10)]
    harmonic_magnitudes = [fft_spectrum[int(f)] if f < len(fft_spectrum) else 0 for f in harmonics]
    
    
    avg_frequency = np.mean(freq_axis)
    avg_amplitude = np.mean(fft_spectrum)
    
    
    avg_frequency_var.set(f'Average Frequency: {avg_frequency:.2f} Hz')
    avg_amplitude_var.set(f'Average Amplitude: {avg_amplitude:.2f} units')
    avg_harmonics_var.set(f'Average Harmonics: {np.mean(harmonic_magnitudes):.2f} units')

    return harmonics, harmonic_magnitudes

def update_plots(sound_wave, rate):
    global canvas
    if canvas is None:
        return

   
    ax1.clear()
    ax1.set_title('Waveform', fontsize=12)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    time_axis = np.linspace(0, len(sound_wave) / rate, num=len(sound_wave))
    ax1.plot(time_axis, sound_wave, color='red')

    
    fft_spectrum = np.abs(np.fft.fft(sound_wave))[:len(sound_wave) // 2]
    freq_axis = np.fft.fftfreq(len(sound_wave), 1 / rate)[:len(sound_wave) // 2]
    ax2.clear()
    ax2.set_title('Frequency Spectrum', fontsize=12)
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_ylabel('Magnitude', fontsize=10)
    ax2.plot(freq_axis, fft_spectrum, color='blue')

    
    harmonics, harmonic_magnitudes = compute_harmonics(sound_wave, rate)
    ax3.clear()
    ax3.set_title('Harmonics', fontsize=12)
    ax3.set_xlabel('Frequency (Hz)', fontsize=10)
    ax3.set_ylabel('Magnitude', fontsize=10)
    ax3.plot(harmonics, harmonic_magnitudes, 'o-', color='green')

    
    canvas.draw()

def plot_waveform(file_path=None):
    global fig, ax1, ax2, ax3, canvas
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))
    plt.tight_layout()

    try:
        if file_path:
            wave_file = wave.open(file_path, 'r')
            sample_width = wave_file.getsampwidth()
            frames = wave_file.getnframes()
            rate = wave_file.getframerate()

            if sample_width != 2:
                raise ValueError("Only 16-bit PCM WAV files are supported")

            raw_data = wave_file.readframes(frames)
            sound_wave = np.frombuffer(raw_data, dtype=np.int16)

            
            update_plots(sound_wave, rate)

            
            for widget in plot_frame.winfo_children():
                widget.destroy()

            
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        else:
            for ax in [ax1, ax2, ax3]:
                ax.clear()
    except Exception as e:
        print(f"Error plotting waveform: {e}")
    finally:
        plt.close(fig)

def separate_vocals(file_path):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(file_path, 'output/')
    vocal_file = 'output/' + os.path.basename(file_path).replace('.mp3', '/vocals.wav')
    return vocal_file

def transcribe_vocals(vocal_file):
    speech, sample_rate = sf.read(vocal_file)
    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def synthesize_speech(text, output_file='output_speech.wav'):
    
    return output_file

def modify_and_synthesize():
    song = song_box.get(ACTIVE)
    wav_path = convert_to_wav(song)
    
    try:
        vocal_path = separate_vocals(wav_path)
        transcribed_text = transcribe_vocals(vocal_path)
        
        transcription_entry.delete(0, END)  
        transcription_entry.insert(0, transcribed_text)

        def synthesize():
            modified_text = transcription_entry.get()
            output_speech = synthesize_speech(modified_text)
            pygame.mixer.music.load(output_speech)
            pygame.mixer.music.play(loops=0)

        synthesize_btn = Button(modify_frame, text="Synthesize", command=synthesize, width=15)
        synthesize_btn.pack(pady=5)
    
    except Exception as e:
        print(f"Error modifying and synthesizing: {e}")

def play():
    global paused, stopped  
    if stopped:
        song = song_box.get(ACTIVE)
        if song.endswith(".mp3"):
            pygame.mixer.music.load(song)
        else:
            wav_path = convert_to_wav(song)
            pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(loops=0)
        update_plots_real_time(song)  
        play_time()  
        stopped = False
        paused = False

        
        threading.Thread(target=update_plots_real_time, args=(song,), daemon=True).start()
    else:
        if paused:
            pygame.mixer.music.unpause()
            paused = False


def update_plots_real_time(song):
    wav_path = convert_to_wav(song)
    wave_file = wave.open(wav_path, 'r')
    frames = wave_file.getnframes()
    rate = wave_file.getframerate()
    total_time = frames / rate
    start_time = 0

    while not stopped:
        
        current_position = pygame.mixer.music.get_pos() / 1000 

        
        if current_position >= total_time:
            break

        
        wave_file.setpos(int(rate * current_position))
        sound_wave = wave_file.readframes(rate // 10)  
        sound_wave = np.frombuffer(sound_wave, dtype=np.int16)

        
        update_plots(sound_wave, rate)
        
        
        time.sleep(0.1)

    wave_file.close()

def stop():
    global stopped
    stopped = True
    pygame.mixer.music.stop()

def pause():
    global paused
    if not paused:
        pygame.mixer.music.pause()
        paused = True

def play_time():
    global song_length
    if not stopped:
        current_time = pygame.mixer.music.get_pos() / 1000  
        progress_var.set(current_time)  
        root.after(1000, play_time)  


load_btn = Button(root, text="Load Songs", command=load_songs)
load_btn.pack(pady=10)

song_box = Listbox(root, width=50, height=10)
song_box.pack(pady=10)

play_btn = Button(root, text="Play", command=play)
play_btn.pack(pady=5)

stop_btn = Button(root, text="Stop", command=stop)
stop_btn.pack(pady=5)

pause_btn = Button(root, text="Pause", command=pause)
pause_btn.pack(pady=5)


plot_frame = Frame(root)
plot_frame.pack(pady=10)


modify_frame = Frame(root)
modify_frame.pack(pady=10)

transcription_entry = Entry(modify_frame, width=50)
transcription_entry.pack(pady=5)

modify_btn = Button(modify_frame, text="Modify and Synthesize", command=modify_and_synthesize)
modify_btn.pack(pady=5)


avg_amplitude_label = Label(modify_frame, textvariable=avg_amplitude_var)
avg_amplitude_label.pack(pady=5)

avg_frequency_label = Label(modify_frame, textvariable=avg_frequency_var)
avg_frequency_label.pack(pady=5)

avg_harmonics_label = Label(modify_frame, textvariable=avg_harmonics_var)
avg_harmonics_label.pack(pady=5)


progress_var = StringVar()
progress_label = Label(root, textvariable=progress_var)
progress_label.pack(pady=5)


root.mainloop()
