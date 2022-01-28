"""Adrian Matyszczak - s19850
   Michał Kaczmarek - s18464

   """
import collections
import datetime

import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Częstotliwość próbkowania dla odtwarzania dźwięku
_SAMPLING_RATE = 16000

#pobieranie danych z Maestro
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )
#Zbiór danych zawiera około 1200 plików MIDI.
filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))
#przetworzenie jednego pliku midi
sample_file = filenames[1]
print(sample_file)

#Wygeneruj obiekt PrettyMIDI dla przykładowego pliku MIDI.
pm = pretty_midi.PrettyMIDI(sample_file)

#Odtwórz przykładowy plik. Załadowanie widżetu odtwarzania może potrwać kilka sekund.
def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  # Pobiernie próbkę wygenerowanego przebiegu, aby złagodzić resetowanie jądra
  waveform_short = waveform[:seconds*_SAMPLING_RATE]
  return display.Audio(waveform_short, rate=_SAMPLING_RATE)

#display_audio(pm)

#Zrób trochę inspekcji pliku MIDI. Jakie instrumenty są używane?
print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)

#Wyodrębnij notatki


# Do trenowania modelu uzywamy zmiennych
# pitch, jakość dzwięku jako nr nuty MIDI
# duration(czas jak długo nuta będzie grana w sekundach i jest różnicą medzy czasem zakończenia nuty a czasem rozpoczęcia nuty.)
for i, note in enumerate(instrument.notes[:10]):
  note_name = pretty_midi.note_number_to_name(note.pitch)
  duration = note.end - note.start
  print(f'{i}: pitch={note.pitch}, note_name={note_name},'
        f' duration={duration:.4f}')

#Wyodrębione nuty z przykładowego pliku MIDI.

def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sortuje notatki według czasu rozpoczęcia
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

raw_notes = midi_to_notes(sample_file)
raw_notes.head()

#Interpretacja nazw nut może być łatwiejsza niż ich wysokości, więc możesz użyć poniższej funkcji, aby przekonwertować wartości numeryczne na nazwy nut. Nazwa nuty pokazuje rodzaj nuty, znak chromatyczny i numer oktawy (np. C#4).

get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
sample_note_names[:10]

#Aby zwizualizować utwór muzyczny, wykreślamy wysokość nuty,
# początek i koniec na całej długości ścieżki (tj. piano roll).
# zaczynamy od pierwszych 100 nut

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)

plot_piano_roll(raw_notes, count=100)


plot_piano_roll(raw_notes)

#Sprawdź rozkład każdej zmiennej nuty.

def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
  plt.figure(figsize=[15, 5])
  plt.subplot(1, 3, 1)
  sns.histplot(notes, x="pitch", bins=20)

  plt.subplot(1, 3, 2)
  max_step = np.percentile(notes['step'], 100 - drop_percentile)
  sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

  plt.subplot(1, 3, 3)
  max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
  sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))

plot_distributions(raw_notes)

#Możesz wygenerować własny plik MIDI z listy nut, korzystając z poniższej funkcji.

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # głośność nut
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm
#zapis do pliku
example_file = 'example.midi'
example_pm = notes_to_midi(
    raw_notes, out_file=example_file, instrument_name=instrument_name)

#Odtwórz wygenerowany plik MIDI i sprawdź, czy jest jakaś różnica.
# display_audio(example_pm)

#Utwórz treningowy zbiór danych
num_files = 5
all_notes = []
for f in filenames[:num_files]:
  notes = midi_to_notes(f)
  all_notes.append(notes)

all_notes = pd.concat(all_notes)

n_notes = len(all_notes)
print('Number of notes parsed:', n_notes)

#Następnie utwórz zestaw tf.data.Dataset z przeanalizowanych nut.
key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
notes_ds.element_spec

# Nauczamy model na partiach sekwencji nut. Każdy przykład będzie składał się z sekwencji nut jako funkcji wejściowych,
# a następna nuta jako etykieta. W ten sposób model zostanie wytrenowany do przewidywania następnej nuty w sekwencji.

def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
  """Zwraca zestaw danych TF z przykładami sekwencji i etykiet"""
  seq_length = seq_length+1

  # Take 1 extra
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # flat_map` spłaszcza „zestaw danych zestawów danych” w zestaw danych tensorów
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalizuj wysokość dźwięku
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Dzielenie etykiet
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

#Rozmiar słownika ( vocab_size ) jest ustawiony na 128,
# co oznacza wszystkie tony obsługiwane przez pretty_midi
#seq_length - długość sekwencji dla każdego przydziału
seq_length = 500
vocab_size = 128
seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
seq_ds.element_spec

#model weźmie 100 nut jako dane wejściowe
# i nauczy się przewidywać następną notatkę jako dane wyjściowe.

for seq, target in seq_ds.take(1):
  print('sequence shape:', seq.shape)
  print('sequence elements (first 10):', seq[0: 10])
  print()
  print('target:', target)

#Grupowanie przykłady i knfiguracja zestawow danych pod kątem wydajności.

batch_size = 64
buffer_size = n_notes - seq_length  # liczba pozycji w zbiorze danych
train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))

train_ds.element_spec

# Twórz i trenuj model
# Model będzie miał trzy wyjścia, po jednym dla każdej zmiennej nuty.

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)

input_shape = (seq_length, 3)
learning_rate = 0.005

inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)

outputs = {
  'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
  'step': tf.keras.layers.Dense(1, name='step')(x),
  'duration': tf.keras.layers.Dense(1, name='duration')(x),
}

model = tf.keras.Model(inputs, outputs)

loss = {
      'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True),
      'step': mse_with_positive_pressure,
      'duration': mse_with_positive_pressure,
}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss=loss, optimizer=optimizer)

model.summary()

#Testowanie funkcji model.evaluate,

losses = model.evaluate(train_ds, return_dict=True)
losses

#Jednym ze sposobów zrównoważenia tego jest użycie argumentu loss_weights do kompilacji:
# zmienna loss staje się sumą poszczególnych strat.
model.compile(
    loss=loss,
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration':1.0,
    },
    optimizer=optimizer,
)



model.evaluate(train_ds, return_dict=True)

#Trenuj modele

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
]


epochs = 10

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
)

plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()

#Aby użyć modelu do generowania nut,
# musimy najpierw podać początkową sekwencję nut.
# Poniższa funkcja generuje jedną nutę z sekwencji nut.

# W przypadku wysokości nuty pobiera próbkę z rozkładu softmax nut wygenerowanych przez model
# i nie wybiera po prostu nuty z największym prawdopodobieństwem.
# Zawsze wybieranie nuty z największym prawdopodobieństwem prowadziłoby do generowania powtarzających się sekwencji nut.
def predict_next_note(
    notes: np.ndarray,
    keras_model: tf.keras.Model,
    temperature: float = 1.0) -> int:
  """Generuje identyfikatory nut przy użyciu wytrenowanego modelu sekwencji."""

  assert temperature > 0

  # dodawanie wymiaru partii
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']

  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # Wartości „step” i „duration” nie mogą być ujemne
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)

#Teraz generujemy kilka nut.

temperature = 2.0
num_predictions = 100

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

# Początkowa sekwencja nut; pitch jest znormalizowana do treningu
input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

generated_notes = []
prev_start = 0
for _ in range(num_predictions):
  pitch, step, duration = predict_next_note(input_notes, model, temperature)
  start = prev_start + step
  end = start + duration
  input_note = (pitch, step, duration)
  generated_notes.append((*input_note, start, end))
  input_notes = np.delete(input_notes, 0, axis=0)
  input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
  prev_start = start

generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end'))

generated_notes.head(10)

out_file = 'output.mid'
out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=instrument_name)
# display_audio(out_pm)