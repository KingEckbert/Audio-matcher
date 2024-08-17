import tkinter as tk
from tkinter import Toplevel, messagebox
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import resample
import matplotlib.colors as mcolors

# Global variables
listening = False
uploaded_sound = None
sr_value = None
mute_timer = None
stream = None
uploaded_sound_binary = None
selected_device = None
similarity_threshold = 0.9
silence_threshold = 0.01
rolling_similarity = []

# Metrics
volume_level = 0
pitch_value = 0
pitch_values = []  # To store pitch values over time

# Resize the graphs to be smaller initially
fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
xdata = np.arange(0, 1024)
ydata = np.zeros(1024)
line, = ax.plot(xdata, ydata, lw=0.8)

ax.set_title("Real-Time Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

fig_spectrogram, ax_spectrogram = plt.subplots(figsize=(3, 2))  # Smaller figure size
ax_spectrogram.set_title("Real-Time Audio Spectrogram")
ax_spectrogram.set_xlabel("Time")
ax_spectrogram.set_ylabel("Frequency")

comparison_label = None
volume_label = None
pitch_label = None


def audio_to_binary(audio_data):
    normalized_audio = np.interp(audio_data, (audio_data.min(), audio_data.max()), (0, 65535)).astype(np.uint16)
    binary_audio = np.unpackbits(normalized_audio.view(np.uint8))
    return binary_audio


def is_silence(audio_chunk):
    return np.max(np.abs(audio_chunk)) < silence_threshold


def load_hardcoded_sound():
    global uploaded_sound, sr_value, uploaded_sound_binary
    try:
        sound_file_path = 'topstreamsaudio.wav'  # Replace with your actual file name
        sr_value, uploaded_sound = wavfile.read(sound_file_path)
        uploaded_sound_mono = np.mean(uploaded_sound, axis=1)
        uploaded_sound_binary = audio_to_binary(uploaded_sound_mono)
        print("Hardcoded audio file loaded and converted to binary successfully.")
    except Exception as e:
        print(f"Failed to load the hardcoded audio file: {e}")


def resample_audio(audio_chunk, target_rate, current_rate):
    num_samples = int(len(audio_chunk) * float(target_rate) / current_rate)
    return resample(audio_chunk, num_samples)


def estimate_volume(audio_chunk):
    """Estimate the volume level from the audio chunk."""
    return np.sqrt(np.mean(np.square(audio_chunk)))


def estimate_pitch(audio_chunk, sample_rate):
    """Estimate the pitch (frequency) of the audio chunk."""
    corr = np.correlate(audio_chunk, audio_chunk, mode='full')
    corr = corr[len(corr) // 2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    pitch = sample_rate / peak if peak > 0 else 0
    return pitch


def update_plot(audio_chunk):
    downsample_factor = max(1, len(audio_chunk) // 1024)
    ydata = np.abs(audio_chunk[::downsample_factor])

    if len(ydata) > len(xdata):
        ydata = ydata[:len(xdata)]
    elif len(ydata) < len(xdata):
        ydata = np.pad(ydata, (0, len(xdata) - len(ydata)), 'constant')

    window_size = 20
    ydata_smoothed = np.convolve(ydata, np.ones(window_size) / window_size, mode='same')

    max_amplitude = np.max(ydata_smoothed)
    ax.set_ylim(-max_amplitude * 1.2, max_amplitude * 1.2)

    line.set_ydata(ydata_smoothed)
    canvas_waveform.draw()


def update_spectrogram(audio_chunk):
    ax_spectrogram.clear()
    Pxx, freqs, bins, im = ax_spectrogram.specgram(audio_chunk, NFFT=1024, Fs=sr_value, noverlap=512,
                                                   cmap='inferno', norm=mcolors.PowerNorm(0.5))
    ax_spectrogram.set_title("Real-Time Audio Spectrogram")
    ax_spectrogram.set_xlabel("Time")
    ax_spectrogram.set_ylabel("Frequency")
    canvas_spectrogram.draw()


def compare_binaries(real_time_binary):
    global uploaded_sound_binary
    min_length = min(len(uploaded_sound_binary), len(real_time_binary))
    comparison_result = np.sum(uploaded_sound_binary[:min_length] == real_time_binary[:min_length]) / min_length
    return comparison_result


def audio_callback(indata, frames, time, status):
    global mute_timer, rolling_similarity, volume_level, pitch_value
    try:
        if listening and uploaded_sound_binary is not None:
            buffer_sound_mono = np.mean(indata, axis=1)

            if sr_value is not None and sr_value != stream.samplerate:
                buffer_sound_mono = resample_audio(buffer_sound_mono, sr_value, stream.samplerate)

            update_plot(buffer_sound_mono)
            update_spectrogram(buffer_sound_mono)

            real_time_binary = audio_to_binary(buffer_sound_mono)
            comparison_result = compare_binaries(real_time_binary)
            comparison_label.config(text=f"Binary Comparison: {comparison_result:.2f}")

            # Update volume and pitch metrics
            volume_level = estimate_volume(buffer_sound_mono)
            volume_label.config(text=f"Volume Level: {volume_level:.2f}")
            pitch_value = estimate_pitch(buffer_sound_mono, sr_value)
            pitch_label.config(text=f"Pitch: {pitch_value:.2f} Hz")

            # Store pitch values
            if pitch_value > 0:  # Only store positive pitch values
                pitch_values.append(pitch_value)

            if is_silence(buffer_sound_mono):
                return

    except Exception as e:
        print(f"Error in audio callback: {e}")


def show_pitch_histogram():
    """Display a histogram of the pitch values in a new window."""
    if not pitch_values:
        messagebox.showinfo("No Data", "No pitch data available.")
        return

    new_window = Toplevel(root)
    new_window.title("Pitch Histogram")

    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    ax_hist.hist(pitch_values, bins=20, color='blue', edgecolor='black')
    ax_hist.set_title("Pitch Histogram")
    ax_hist.set_xlabel("Pitch (Hz)")
    ax_hist.set_ylabel("Frequency")

    canvas_histogram = FigureCanvasTkAgg(fig_hist, master=new_window)
    canvas_histogram.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    canvas_histogram.draw()


def mute_audio():
    print("Muting audio...")


def unmute_audio():
    print("Unmuting audio")


def start_listening():
    global listening, stream, rolling_similarity, selected_device
    if uploaded_sound_binary is None:
        messagebox.showwarning("No File", "Failed to load the hardcoded audio file.")
        return
    listening = True
    rolling_similarity = []
    if stream is None or not stream.active:
        stream = sd.InputStream(callback=audio_callback, device=selected_device, channels=1, samplerate=sr_value,
                                blocksize=1024)
    stream.start()
    print(f'Listening on device {selected_device}...')


def stop_listening():
    global listening, stream, mute_timer
    listening = False
    if mute_timer is not None:
        mute_timer.cancel()
        mute_timer = None
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
    print(f'Not listening...')


def update_device_selection(event):
    global selected_device
    selected_device = int(device_listbox.get(device_listbox.curselection()).split(":")[0])
    print(f"Selected device: {selected_device}")


def show_full_size_graph(fig, title):
    new_window = Toplevel(root)
    new_window.title(title)

    full_fig, full_ax = plt.subplots(figsize=(8, 6))
    new_canvas = FigureCanvasTkAgg(fig, master=new_window)
    new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    new_canvas.draw()


# Load the hardcoded sound file when the program starts
load_hardcoded_sound()

# GUI
root = tk.Tk()
root.title("Sound Listener")

frame_waveform = tk.Frame(root)
frame_spectrogram = tk.Frame(root)
frame_waveform.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=5)
frame_spectrogram.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=5)

canvas_waveform = FigureCanvasTkAgg(fig, master=frame_waveform)
canvas_waveform.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas_waveform.mpl_connect("button_press_event", lambda event: show_full_size_graph(fig, "Waveform"))

canvas_spectrogram = FigureCanvasTkAgg(fig_spectrogram, master=frame_spectrogram)
canvas_spectrogram.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas_spectrogram.mpl_connect("button_press_event", lambda event: show_full_size_graph(fig_spectrogram, "Spectrogram"))

# Information Frame
info_frame = tk.Frame(root)
info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

comparison_label = tk.Label(info_frame, text="Binary Comparison: ", font=("Helvetica", 12))
comparison_label.grid(row=0, column=0, padx=10, pady=5)

volume_label = tk.Label(info_frame, text="Volume Level: 0.00", font=("Helvetica", 12))
volume_label.grid(row=1, column=0, padx=10, pady=5)

pitch_label = tk.Label(info_frame, text="Pitch: 0.00 Hz", font=("Helvetica", 12))
pitch_label.grid(row=2, column=0, padx=10, pady=5)

device_label = tk.Label(root, text="Select Audio Input Device:", font=("Helvetica", 12))
device_label.pack(pady=5)

device_listbox = tk.Listbox(root, height=6, font=("Helvetica", 10))
for device in [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]:
    device_listbox.insert(tk.END, device)
device_listbox.pack(pady=5)

device_listbox.bind("<<ListboxSelect>>", update_device_selection)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Listening", command=start_listening, font=("Helvetica", 12),
                         width=15)
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(button_frame, text="Stop Listening", command=stop_listening, font=("Helvetica", 12), width=15)
stop_button.grid(row=0, column=1, padx=10)

# Button to show pitch histogram
histogram_button = tk.Button(button_frame, text="Show Pitch Histogram", command=show_pitch_histogram,
                             font=("Helvetica", 12), width=20)
histogram_button.grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()
