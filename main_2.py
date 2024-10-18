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
import struct
from pydub import AudioSegment
import os
import threading

root = Tk()
root.title('Sound Dynamiction Cross X')
root.geometry("800x800")


pygame.mixer.init()

def convert_to_wav(file_path):
    print(f"กำลังแปลงไฟล์: {file_path}")  
    if file_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace(".mp3", ".wav")
        print(f"Exporting to WAV: {wav_path}")  
        sound.export(wav_path, format="wav")
        print(f"ไฟล์ WAV ที่แปลงแล้ว: {wav_path}")  
        return wav_path
    return file_path

def compute_harmonics(sound_wave, rate):
    fft_spectrum = np.abs(np.fft.fft(sound_wave))[:len(sound_wave) // 2]
    freq_axis = np.fft.fftfreq(len(sound_wave), 1 / rate)[:len(sound_wave) // 2]
    fundamental_freq = np.argmax(fft_spectrum)
    harmonics = [fundamental_freq * (i + 1) for i in range(10)]
    harmonic_magnitudes = [fft_spectrum[int(f)] if f < len(fft_spectrum) else 0 for f in harmonics]

    return harmonics, harmonic_magnitudes

def plot_waveform(file_path=None):
    print(f"กำลังพล็อตกราฟสำหรับไฟล์: {file_path}")
    global wave_file, sound_wave, time_axis, frames, rate, anim, canvas, freq_anim, harmonic_anim
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 2.5))  

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveform')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Spectrum')

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Harmonics')

    try:
        if file_path:
            wave_file = wave.open(file_path, 'r')
            sample_width = wave_file.getsampwidth()
            frames = wave_file.getnframes()
            rate = wave_file.getframerate()
            duration = frames / float(rate)

            if sample_width != 2:
                raise ValueError("Only 16-bit PCM WAV files are supported")

            
            raw_data = wave_file.readframes(frames)
            sound_wave = np.frombuffer(raw_data, dtype=np.int16)

            
            max_frames = min(len(sound_wave), int(rate * 240))
            sound_wave = sound_wave[:max_frames]
            time_axis = np.linspace(0, len(sound_wave) / rate, num=len(sound_wave))

            
            fft_spectrum = np.abs(np.fft.fft(sound_wave))[:len(sound_wave) // 2]
            freq_axis = np.fft.fftfreq(len(sound_wave), 1 / rate)[:len(sound_wave) // 2]
            harmonics, harmonic_magnitudes = compute_harmonics(sound_wave, rate)

            
            print(f"Waveform Data (First 10 samples): {sound_wave[:10]}")
            print(f"FFT Spectrum (First 10 values): {fft_spectrum[:10]}")
            print(f"Harmonics: {harmonics}")
            print(f"Harmonic Magnitudes: {harmonic_magnitudes}")

           
            anim, = ax1.plot(time_axis, sound_wave, color='red', label='Amplitude')
            ax1.set_xlim(0, min(duration, 240))
            ax1.set_ylim(min(sound_wave), max(sound_wave))
            ax1.legend()

            
            freq_anim, = ax2.plot(freq_axis, fft_spectrum, color='blue', label='Frequency Spectrum')
            ax2.set_xlim(0, 1000)
            ax2.set_ylim(0, np.max(fft_spectrum))
            ax2.legend()

            
            harmonic_anim, = ax3.plot(harmonics, harmonic_magnitudes, 'o', color='green', label='Harmonics')
            ax3.set_xlim(0, max(harmonics) * 1.2)
            ax3.set_ylim(0, np.max(harmonic_magnitudes))
            ax3.legend()
        else:
            for ax in [ax1, ax2, ax3]:
                ax.clear()

        for widget in plot_frame.winfo_children():
            widget.destroy()

        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    except Exception as e:
        print(f"Error plotting waveform: {e}")
    finally:
        plt.close(fig)


def load_and_plot(file_path):
    """Handles file loading and plotting in a separate thread."""
    threading.Thread(target=plot_waveform, args=(file_path,), daemon=True).start()

def update_plot():
    if pygame.mixer.music.get_busy():
        current_time = pygame.mixer.music.get_pos() / 1000
        index = int(current_time * rate)
        if index < len(sound_wave):
            anim.set_ydata(sound_wave[:index])
            canvas.draw()
        root.after(50, update_plot)  

def play_time():
    global song_length
    if stopped:
        return
    current_time = pygame.mixer.music.get_pos() / 1000
    converted_current_time = time.strftime('%M:%S', time.gmtime(current_time))
    song = song_box.get(ACTIVE)
    if song.endswith(".mp3"):
        song_mut = MP3(song)
        song_length = song_mut.info.length
    else:
        wave_file = wave.open(song, 'r')
        frames = wave_file.getnframes()
        rate = wave_file.getframerate()
        song_length = frames / float(rate)
        wave_file.close()

    converted_song_length = time.strftime('%M:%S', time.gmtime(song_length))
    current_time += 1
    if int(my_slider.get()) == int(song_length):
        status_bar.config(text=f'Time Elapsed: {converted_song_length}  of  {converted_song_length}  ')
    elif paused:
        pass
    elif int(my_slider.get()) == int(current_time):
        slider_position = int(song_length)
        my_slider.config(to=slider_position, value=int(current_time))
    else:
        slider_position = int(song_length)
        my_slider.config(to=slider_position, value=int(my_slider.get()))
        converted_current_time = time.strftime('%M:%S', time.gmtime(int(my_slider.get())))
        status_bar.config(text=f'Time Elapsed: {converted_current_time}  of  {converted_song_length}  ')
        next_time = int(my_slider.get()) + 1
        my_slider.config(value=next_time)
    status_bar.after(1000, play_time)

def add_song():
    song = filedialog.askopenfilename(initialdir='audio/', title="Choose A Song", filetypes=(("Audio Files", "*.mp3 *.wav"), ))
    song_box.insert(END, song)

def add_many_songs():
    songs = filedialog.askopenfilenames(initialdir='audio/', title="Choose A Song", filetypes=(("Audio Files", "*.mp3 *.wav"), ))
    for song in songs:
        song_box.insert(END, song)

def update_averages(sound_wave, rate):
    
    fft_spectrum = np.abs(np.fft.fft(sound_wave))[:len(sound_wave) // 2]
    freq_axis = np.fft.fftfreq(len(sound_wave), 1 / rate)[:len(sound_wave) // 2]
    
    avg_frequency = np.mean(freq_axis)
    avg_amplitude = np.mean(np.abs(sound_wave))

    
    harmonics, harmonic_magnitudes = compute_harmonics(sound_wave, rate)
    avg_harmonics = np.mean(harmonic_magnitudes)
    
    return avg_frequency, avg_amplitude, avg_harmonics

def play():
    global stopped
    stopped = False
    song = song_box.get(ACTIVE)

    try:
        print(f"กำลังโหลดเพลง: {song}")  
        wav_path = convert_to_wav(song)
        print(f"เส้นทางของไฟล์ WAV ที่แปลง: {wav_path}")  

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"ไม่พบไฟล์ WAV: {wav_path}")
        load_and_plot(wav_path)

        
        wave_file = wave.open(wav_path, 'r')
        frames = wave_file.getnframes()
        rate = wave_file.getframerate()
        raw_data = wave_file.readframes(frames)
        sound_wave = np.frombuffer(raw_data, dtype=np.int16)

        avg_frequency, avg_amplitude, avg_harmonics = update_averages(sound_wave, rate)

       
        avg_label.config(text=f"Avg Frequency: {avg_frequency:.2f} Hz | "
                              f"Avg Amplitude: {avg_amplitude:.2f} | "
                              f"Avg Harmonics: {avg_harmonics:.2f}",
                         bg='black', fg='white')

        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(loops=0)
        play_time()
    except Exception as e:
        print(f"เกิดข้อผิดพลาดขณะเล่นเพลง: {e}")


global stopped
stopped = False

def stop():
    status_bar.config(text='')
    my_slider.config(value=0)
    pygame.mixer.music.stop()
    song_box.selection_clear(ACTIVE)
    status_bar.config(text='')
    global stopped
    stopped = True
    plot_waveform()  

def next_song():
    status_bar.config(text='')
    my_slider.config(value=0)
    next_one = song_box.curselection()
    next_one = next_one[0] + 1
    song = song_box.get(next_one)

    wav_path = convert_to_wav(song)
    plot_waveform(wav_path)

    pygame.mixer.music.load(wav_path)
    pygame.mixer.music.play(loops=0)
    song_box.selection_clear(0, END)
    song_box.activate(next_one)
    song_box.selection_set(next_one, last=None)

    if song.endswith(".mp3"):
        os.remove(wav_path)

def previous_song():
    status_bar.config(text='')
    my_slider.config(value=0)
    next_one = song_box.curselection()
    next_one = next_one[0] - 1
    song = song_box.get(next_one)

    wav_path = convert_to_wav(song)
    plot_waveform(wav_path)

    pygame.mixer.music.load(wav_path)
    pygame.mixer.music.play(loops=0)
    song_box.selection_clear(0, END)
    song_box.activate(next_one)
    song_box.selection_set(next_one, last=None)

    if song.endswith(".mp3"):
        os.remove(wav_path)

def delete_song():
    stop()
    song_box.delete(ANCHOR)
    pygame.mixer.music.stop()

def delete_all_songs():
    stop()
    song_box.delete(0, END)
    pygame.mixer.music.stop()

global paused
paused = False

def pause(is_paused):
    global paused
    paused = is_paused
    if paused:
        pygame.mixer.music.unpause()
        paused = False
    else:
        pygame.mixer.music.pause()
        paused = True

def slide(x):
    song = song_box.get(ACTIVE)
    if song:
        pygame.mixer.music.load(song)
        pygame.mixer.music.play(loops=0, start=int(my_slider.get()))

def volume(x):
    pygame.mixer.music.set_volume(volume_slider.get())
    current_volume = pygame.mixer.music.get_volume() * 100
    if int(current_volume) < 1:
        volume_meter.config(image=vol0)
    elif int(current_volume) > 0 and int(current_volume) <= 25:
        volume_meter.config(image=vol1)
    elif int(current_volume) >= 25 and int(current_volume) <= 50:
        volume_meter.config(image=vol2)
    elif int(current_volume) >= 50 and int(current_volume) <= 75:
        volume_meter.config(image=vol3)
    elif int(current_volume) >= 75 and int(current_volume) <= 100:
        volume_meter.config(image=vol4)


master_frame = Frame(root)
master_frame.pack(pady=20)

song_box = Listbox(master_frame, bg="black", fg="green", width=60, selectbackground="gray", selectforeground="black")
song_box.grid(row=0, column=0)

controls_frame = Frame(master_frame)
controls_frame.grid(row=1, column=0, pady=20)

back_btn_img = PhotoImage(file='images/back50.png')
forward_btn_img = PhotoImage(file='images/forward50.png')
play_btn_img = PhotoImage(file='images/play50.png')
pause_btn_img = PhotoImage(file='images/pause50.png')
stop_btn_img = PhotoImage(file='images/stop50.png')

vol0 = PhotoImage(file='images/volume0.png')
vol1 = PhotoImage(file='images/volume1.png')
vol2 = PhotoImage(file='images/volume2.png')
vol3 = PhotoImage(file='images/volume3.png')
vol4 = PhotoImage(file='images/volume4.png')

back_button = Button(controls_frame, image=back_btn_img, borderwidth=0, command=previous_song)
forward_button = Button(controls_frame, image=forward_btn_img, borderwidth=0, command=next_song)
play_button = Button(controls_frame, image=play_btn_img, borderwidth=0, command=play)
pause_button = Button(controls_frame, image=pause_btn_img, borderwidth=0, command=lambda: pause(paused))
stop_button = Button(controls_frame, image=stop_btn_img, borderwidth=0, command=stop)

back_button.grid(row=0, column=0, padx=10)
forward_button.grid(row=0, column=1, padx=10)
play_button.grid(row=0, column=2, padx=10)
pause_button.grid(row=0, column=3, padx=10)
stop_button.grid(row=0, column=4, padx=10)

volume_frame = LabelFrame(master_frame, text="Volume")
volume_frame.grid(row=0, column=1, padx=30)

volume_meter = Label(volume_frame, image=vol0)
volume_meter.grid(row=0, column=2)

volume_slider = ttk.Scale(volume_frame, from_=0, to=1, orient=VERTICAL, value=1, command=volume, length=125)
volume_slider.grid(row=0, column=1, padx=20)

my_slider = ttk.Scale(master_frame, from_=0, to=100, orient=HORIZONTAL, value=0, length=360, command=slide)
my_slider.grid(row=2, column=0, pady=10)

status_bar = Label(root, text='', bd=1, relief=GROOVE, anchor=E)
status_bar.pack(fill=X, side=BOTTOM, ipady=2)

avg_label = Label(root, text='', font=('Helvetica', 12), bg='black', fg='white')
avg_label.pack(fill=X, padx=10, pady=10)

plot_frame = Frame(root, width=800, height=500)
plot_frame.pack(pady=20, fill=BOTH, expand=True)

plot_waveform()  

my_menu = Menu(root)
root.config(menu=my_menu)

add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Add Songs", menu=add_song_menu)
add_song_menu.add_command(label="Add One Song To Playlist", command=add_song)
add_song_menu.add_command(label="Add Many Songs To Playlist", command=add_many_songs)

remove_song_menu = Menu(my_menu)
my_menu.add_cascade(label="Remove Songs", menu=remove_song_menu)
remove_song_menu.add_command(label="Delete A Song From Playlist", command=delete_song)
remove_song_menu.add_command(label="Delete All Songs From Playlist", command=delete_all_songs)

root.mainloop()
