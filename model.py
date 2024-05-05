import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from input import Input


class Model:

    def __init__(self, data: tuple[tensorflow.data.Dataset, tensorflow.data.Dataset] | None = None) -> None:
        '''

        Parameters
        ----------
        data
        '''
        self._model = keras.Sequential()
        self._model.add(
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self._model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self._model.add(Flatten())
        self._model.add(Dense(128, activation='relu'))
        self._model.add(Dropout(0.3))
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dense(59, activation='softmax'))
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if data:
            self._train, self._val = data
        else:
            self._train, self._val = None, None
        self._train_history = None

    def train_model(self, n_epochs: int, ignore_save_history: bool = 0) -> None:
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

    def make_plot(self, train_history) -> None:
        '''

        Parameters
        ----------
        train_history

        Returns
        -------

        '''
        plt.plot(self._train_history['loss'])
        plt.plot(self._train_history['val_loss'])
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
        return self._model.predict(np.reshape(input, (1, 28, 28, 1)))

    def img_to_str(self, path: str) -> str:
        '''

        Parameters
        ----------
        path

        Returns
        -------

        '''
        alph = {
            0: 'а',  # 43 английская
            1: 'й',
            2: 'к',  # 45 английская
            3: 'л',
            4: 'н',  # 67 английская
            5: 'о',
            6: 'з',
            7: 'п',
            8: 'м',
            9: 'р',  # 50 английская
            10: 'с',  # 62 английская
            11: 'б',
            12: 'т',  # 55 английская
            13: 'у',  # 60 английская
            14: 'ф',
            15: 'х',  # 59 английская
            16: 'ц',
            17: 'ч',
            18: 'ш',
            19: 'щ',
            20: 'ъ',
            21: 'ы',
            22: 'в',  # 54 английская
            23: 'ь',
            24: 'э',
            25: 'ю',
            26: 'я',
            27: 'г',
            28: 'д',
            29: 'е',  # 64 английская
            30: 'ё',
            31: 'ж',
            32: 'и',  # 56 английская
            33: '0',
            34: '1',
            35: '2',
            36: '3',
            37: '4',
            38: '5',
            39: '6',
            40: '7',
            41: '8',
            42: '9',
            43: 'a',  # 0 русская
            44: 'j',
            45: 'k',  # 2 русская
            46: 'l',
            47: 'm',  # 8 русская
            48: 'n',
            49: 'o',  # 5 русская
            50: 'p',  # 9 русская,
            51: 'q',
            52: 'r',
            53: 's',
            54: 'b',  # 22 русская
            55: 't',  # 12 русская
            56: 'u',  # 32 русская
            57: 'v',
            58: 'w',
            59: 'x',  # 15 русская
            60: 'y',  # 13 русская
            61: 'z',
            62: 'c',  # 10 русская
            63: 'd',
            64: 'e',  # 29 русская
            65: 'f',
            66: 'g',
            67: 'h',  # 4 русская
            68: 'i',
        }
        trouble_pairs = [[0, 43], [2, 45], [4, 67], [9, 50], [10, 62], [12, 55], [8, 47],
                         [13, 60], [15, 59], [22, 54], [29, 64], [32, 56]]
        trouble = [j for i in trouble_pairs for j in i]
        rus = [i for i in range(33)]
        numbs = [i for i in range(33, 43)]
        en = [i for i in range(43, 69)]
        answer = []
        structure = [0, 0, 0]
        letters = Input.get_letters(path)
        for letter in letters:
            pred = self.predict(letter)[0]
            max_el, max_idx = 0, -1
            for i, el in enumerate(pred):
                if el > max_el:
                    max_el = el
                    max_idx = i
            answer.append(max_idx)
            res = int((max_idx in rus)) * 0 + int((max_idx in numbs)) * 1 + int((max_idx in en)) * 2
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
