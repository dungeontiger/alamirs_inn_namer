import io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

STEP = 1
SEQUENCE_LEN = 2
MIN_WORD_FREQUENCY = 1
BATCH_SIZE = 32

# globals...sloppy
examples_file = None
words = set()
sentences = []

def main():
  # read in the samples
  with io.open('train.txt', encoding='utf-8') as f:
      text = f.read().lower().replace('\n', ' . ')
  print('Corpus length in characters:', len(text))

  # split on words including the \n as a sentence
  text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
  print('Corpus length in words:', len(text_in_words))

  # get the unique words
  words = sorted(set(text_in_words))
  print('Unique words:', len(words))

  # create the look ups
  word_indices = dict((c, i) for i, c in enumerate(words))
  indices_word = dict((i, c) for i, c in enumerate(words))

  # arrange into words and next words
  sentences = []
  next_words = []

  for i in range(0, len(text_in_words), STEP):
    # we don't want to predict what comes after a . so filter out those
    a = text_in_words[i: i + SEQUENCE_LEN]
    if '.' not in a:
      sentences.append(a)
      next_words.append(text_in_words[i + SEQUENCE_LEN])

  # split and shuffle into 98% training and 2% testing
  (sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)

  # build the model
  model = Sequential()
  model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
  dropout = 0.2
  if dropout > 0:
    model.add(Dropout(dropout))
  model.add(Dense(len(words)))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

  file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % (
        len(words),
        SEQUENCE_LEN,
        MIN_WORD_FREQUENCY)
  checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
  print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
  early_stopping = EarlyStopping(monitor='val_acc', patience=5)
  callbacks_list = [checkpoint, print_callback, early_stopping]

  examples = 'examples.txt'
  examples_file = open(examples, "w")
  model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                      steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                      epochs=100,
                      callbacks=callbacks_list,
                      validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                      validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)  

def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
  # shuffle at unison
  print('Shuffling sentences')

  tmp_sentences = []
  tmp_next_word = []
  for i in np.random.permutation(len(sentences_original)):
    tmp_sentences.append(sentences_original[i])
    tmp_next_word.append(next_original[i])

  cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
  x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
  y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

  print("Size of training set = %d" % len(x_train))
  print("Size of test set = %d" % len(y_test))
  return (x_train, y_train), (x_test, y_test)

def on_epoch_end(epoch, logs):
  # Function invoked at end of each epoch. Prints generated text.
  examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

  # Randomly pick a seed sequence
  seed_index = np.random.randint(len(sentences + sentences_test))
  seed = (sentences + sentences_test)[seed_index]

  for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
    sentence = seed
    examples_file.write('----- Diversity:' + str(diversity) + '\n')
    examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
    examples_file.write(' '.join(sentence))

    for i in range(50):
      x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
      for t, word in enumerate(sentence):
        x_pred[0, t, word_indices[word]] = 1.

      preds = model.predict(x_pred, verbose=0)[0]
      next_index = sample(preds, diversity)
      next_word = indices_word[next_index]

      sentence = sentence[1:]
      sentence.append(next_word)

      examples_file.write(" "+next_word)
    examples_file.write('\n')
  examples_file.write('='*80 + '\n')
  examples_file.flush()

# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
  # helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
  index = 0
  while True:
    x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
    y = np.zeros((batch_size, len(words)), dtype=np.bool)
    for i in range(batch_size):
      for t, w in enumerate(sentence_list[index % len(sentence_list)]):
        x[i, t, word_indices[w]] = 1
      y[i, word_indices[next_word_list[index % len(sentence_list)]]] = 1
      index = index + 1
    yield x, y

# kick it off
main()
