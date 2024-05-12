import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from input import Input


class Model:

    def __init__(self, new_model: bool = False, model_language_mode: str = 'both_languages', data: tuple[tensorflow.data.Dataset, tensorflow.data.Dataset] | None = None) -> None:
        '''

        Parameters
        ----------
        data
        new_model
        model_language_mode
        '''
        if not new_model:
            languages = {
                'both_languages': 'all_symbols_both_languages_model.h5',
                'cyrillic': 'all_symbols_cyrillic_model.h5',
                'latin': 'all_symbols_latin_model.h5'
            }
            self._model = load_model(languages[model_language_mode])
        else:
            self._model = keras.Sequential()
            self._model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
            self._model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
            self._model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            self._model.add(Flatten())
            self._model.add(Dense(128, activation='relu'))
            self._model.add(Dropout(0.3))
            self._model.add(Dense(256, activation='relu'))
            self._model.add(Dense(69, activation='softmax'))
            self._model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if data:
            self._train, self._val = data
        else:
            self._train, self._val = None, None
        self._train_history = None

    def train_model(self, n_epochs: int = 10, ignore_save_history: bool = 0) -> None:
        '''

        Parameters
        ----------
        n_epochs
        ignore_save_history

        Returns
        -------

        '''
        if not self._train:
            raise ValueError('no data to train')
        model_training_history = self._model.fit(self._train, self._val, epochs=n_epochs).history
        if not ignore_save_history:
            self._train_history = model_training_history

    def make_plot(self, train_history = self._train_history) -> None:
        '''

        Parameters
        ----------
        train_history

        Returns
        -------

        '''
        plt.plot(train_history['loss'])
        plt.plot(train_history['val_loss'])
        plt.grid(True)
        plt.show()

    def save_weights(self, weights_path: str) -> None:
        '''

        Parameters
        ----------
        weights_path

        Returns
        -------

        '''
        self._model.save_weights(weights_path)

    def save_model(self, model_path: str) -> None:
        '''

        Parameters
        ----------
        model_path

        Returns
        -------

        '''
        self._model.save(model_path)

    def load_model(self, model_path: str) -> None:
        '''

        Parameters
        ----------
        model_path

        Returns
        -------

        '''
        self._model = load_model(model_path)

    def load_weights(self, weights_path: str) -> None:
        '''

        Parameters
        ----------
        weights_path

        Returns
        -------

        '''
        self._model.load_weights(weights_path)

    def predict(self, input: np.ndarray) -> int:
        '''

        Parameters
        ----------
        input

        Returns
        -------

        '''
        return np.argmax(self._model.predict(np.reshape(input, (1, 28, 28, 1))))

    def img_to_str(self, path: str) -> str:
        '''

        Parameters
        ----------
        path

        Returns
        -------

        '''
        alph = {
            10: 'а',  # 43 английская
            19: 'и',
            20: 'й',
            21: 'к',  # 53 английская
            22: 'л',
            23: 'м',  # 55 английская
            24: 'н',  # 50 английская
            25: 'о',  # 57 английская
            26: 'п',
            27: 'р',  # 58 английская
            28: 'с',  # 45 английская
            11: 'б',
            29: 'т',  # 62 английская
            30: 'у',  # 67 английская
            31: 'ф',
            32: 'х',  # 66 английская
            33: 'ц',
            34: 'ч',
            35: 'ш',
            36: 'щ',
            37: 'ъ',
            38: 'ы',
            12: 'в',  # 44 английская
            39: 'ь',
            40: 'э',
            41: 'ю',
            42: 'я',
            13: 'г',
            14: 'д',  # 46 английская
            15: 'е',  # 47 английская
            16: 'ё',
            17: 'ж',
            18: 'з',
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            43: 'a',  # 10 русская
            52: 'j',
            53: 'k',  # 21 русская
            54: 'l',
            55: 'm',  # 23 русская
            56: 'n',
            57: 'o',  # 25 русская
            58: 'p',  # 27 русская
            59: 'q',
            60: 'r',
            61: 's',
            44: 'b',  # 12 русская
            62: 't',  # 29 русская
            63: 'u',
            64: 'v',
            65: 'w',
            66: 'x',  # 32 русская
            67: 'y',  # 30 русская
            68: 'z',
            45: 'c',  # 28 русская
            46: 'd',  # 14 русская
            47: 'e',  # 15 русская
            48: 'f',
            49: 'g',
            50: 'h',  # 24 русская
            51: 'i'
        }
        trouble_pairs = [[10, 43], [21, 53], [23, 55], [24, 50], [25, 57], [27, 58], [28, 45],
                         [29, 62], [30, 67], [32, 66], [12, 44], [14, 46], [15, 47]]
        trouble = [j for i in trouble_pairs for j in i]
        numbs = [i for i in range(10)]
        rus = [i for i in range(10, 43)]
        en = [i for i in range(43, 69)]
        answer = []
        structure = [0, 0, 0]
        letters = Input.get_letters(path)
        for letter in letters:
            prediction = self.predict(letter)
            answer.append(prediction)
            res = (int((prediction in numbs)) * 0 + int((prediction in rus)) * 1 + int((prediction in en)) * 2)
            structure[res] += 1
        language = [rus, numbs, en][structure.index(max(structure))]
        for l in range(len(answer)):
            if answer[l] in trouble:
                for pair in trouble_pairs:
                    if answer[l] in pair:
                        if answer[l] not in language:
                            answer[l] = pair[int(not (pair.index(answer[l])))]
            answer[l] = alph[answer[l]]
        return ''.join(answer)
