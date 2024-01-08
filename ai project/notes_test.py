import h5py
import numpy as np
import cupy as cp
from DL7 import *
from matplotlib import pyplot as plt
import librosa
from midiutil.MidiFile import MIDIFile

f = h5py.File(r"C:\Users\omerg\source\repos\note sample generation\note sample generation\note_samples_808.hdf5", "r")
X_total = []
Y_total = np.array([])

for note in range(len(list(f.keys()))):
    dset = np.array(f[list(f.keys())[note]])
    X_total = dset if len(X_total) == 0 else np.concatenate((X_total, dset))
    Y_total = np.append(Y_total, [note] * 58)

X_total = cp.fft.rfft(cp.array(X_total))
X_total_real = cp.real(X_total)
X_total_imag = cp.imag(X_total)
X_total = np.concatenate((X_total_real, X_total_imag), axis=-1)
X_total = cp.asnumpy(X_total)
np.random.seed(1)
np.random.shuffle(X_total)
np.random.seed(1)
np.random.shuffle(Y_total)

X_train = X_total[0:400]
X_train = np.array(X_train).T
Y_train = Y_total[0:400]
Y_train = DLModel.to_one_hot(12, Y_train)
X_test = X_total[400:]
X_test = np.array(X_test).T
Y_test = Y_total[400:]
Y_test = DLModel.to_one_hot(12, Y_test)
n = X_train.shape[0]

X_train = cp.array(X_train)
Y_train = cp.array(Y_train)
X_test = cp.array(X_test)
Y_test = cp.array(Y_test)

#audio, sr = librosa.load(r"C:\Users\omerg\Favorites\Downloads\bass.wav", mono=False, sr=None)
#audio, sr = librosa.load(r"C:\Users\omerg\Desktop\Music\FL\Python Audio Test Samples\808s\808 11 (C).wav", mono=False, sr=None)
#audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=5)
#audio = librosa.to_mono(audio)
#audio = np.array(audio)
hop_size = 0.5
hop_distance = int(66150*hop_size)
hop_time = 1.5
np.random.seed(1)
cp.random.seed(1)
model = DLModel(use_cuda=True)
model.add(DLLayer("L1", 48, (n,), "relu", "He", 0.02, "adam"))
model.add(DLLayer("L3", 12, (48,), "trim_softmax", "Xavier", 0.02, "adam", "L2"))
model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 300, 100)
#model.save_weights("808s/fourier", True)
plt.plot(costs)
plt.show()

print("Train:")
model.confusion_matrix(X_train, Y_train)
print("Test:")
model.confusion_matrix(X_test, Y_test)

'''arr = []
mf = MIDIFile(1)
track = 0
time = 0    
bpm = 148
mf.addTrackName(track, time, "Output")
mf.addTempo(track, time, bpm)
length = bpm/60*hop_time*hop_size

for i in range(int(audio.shape[0]/hop_distance)):
    try:
        s = 0
        for j in range(10):
            s += model.predict(audio[hop_distance*i+j*1000:66150+hop_distance*i+j*1000].reshape((66150,1))).argmax()
        arr.append(int(s/10))
    except:
        break
        

for i in range(len(arr)):
    mf.addNote(track, 0, 57+arr[i] if arr[i] > 2 else 69+arr[i], i*length, length, 100)

samples_per_beat = audio.shape[0]/(bpm*3.54)


for i in range(audio.shape[0]):
    if np.abs(audio[i]) < 1e-4:
        note = model.predict(audio[i:66150+i].reshape((66150,1))).argmax()
        mf.addNote(track, 0, 57+note if note > 2 else 69+note, int(i/samples_per_beat), 1, 100)

with open("output.mid", 'wb') as outf:
    mf.writeFile(outf)

#print(arr)
#plt.plot(arr)
#plt.show()'''