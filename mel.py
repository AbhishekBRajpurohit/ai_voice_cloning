import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio
audio, sr = librosa.load("Recording.wav")

# Create Mel Spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr
)

# Convert to decibels
mel_db = librosa.power_to_db(mel, ref=mel.max())

# Display Mel Spectrogram
librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()
