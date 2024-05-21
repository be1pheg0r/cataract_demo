import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from input import Input


class Model:

    def __init__(self, new_model: bool = False, model_language_mode: str = 'both_languages') -> None:
        '''

        Parameters
        ----------
        new_model
        model_language_mode

        Return
        ----------
        '''
        if not new_model:
            if model_language_mode == 'both_languages':
                self.__letter_processing_needed = True
            else:
                self.__letter_processing_needed = False
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
            self.__letter_processing_needed = False
        self._train_history = None


    def train_model(self, train_data: tensorflow.data.Dataset, val_data: tensorflow.data.Dataset, n_epochs: int = 10, ignore_save_history: bool = False) -> None:
        '''

        Parameters
        ----------
        n_epochs
        ignore_save_history
        train_data
        val_data

        Returns
        -------

        '''
        model_training_history = self._model.fit(train_data, val_data, epochs=n_epochs).history
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
        int
        '''
        return np.argmax(self._model.predict(np.reshape(input, (1, 28, 28, 1))))

    def img_to_str(self, path: str) -> str:
        '''

        Parameters
        ----------
        path

        Returns
        -------
        str
        '''

        def determine_the_expected_language_is_cyrillic(number_of_trouble_cyrillic_letters: int, number_of_not_trouble_cyrillic_letters: int, number_of_trouble_latin_letters: int, number_of_not_trouble_latin_letters: int) -> bool:
            '''
            Parameters
            ----------
            number_of_trouble_cyrillic_letters
            number_of_not_trouble_cyrillic_letters
            number_of_trouble_latin_letters
            number_of_not_trouble_latin_letters

            Returns
            -------
            bool
            '''
            max_not_trouble_letters = max(number_of_not_trouble_latin_letters, number_of_not_trouble_cyrillic_letters)
            if max_not_trouble_letters != 0:
                if number_of_not_trouble_cyrillic_letters >= number_of_not_trouble_latin_letters:
                    return True
                else:
                    return False
            else:
                if number_of_trouble_cyrillic_letters >= number_of_trouble_latin_letters:
                    return True
                else:
                    return False

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
        trouble_pairs = {10: 43,
                         43: 10,
                         21: 53,
                         53: 21,
                         23: 55,
                         55: 23,
                         24: 50,
                         50: 24,
                         25: 57,
                         57: 25,
                         27: 58,
                         58: 27,
                         28: 45,
                         45: 28,
                         29: 62,
                         62: 29,
                         30: 67,
                         67: 30,
                         32: 66,
                         66: 32,
                         12: 44,
                         44: 12,
                         14: 46,
                         46: 14,
                         15: 47,
                         47: 15
                         }
        trouble_indexes = [10, 12, 14, 15, 21, 23, 24, 25, 27, 28, 29, 30, 32, 43, 44, 45, 46, 47, 50, 53, 55, 57, 58,
                           62, 66, 67]
        cyrillic_index = set(range(10, 43))
        latin_index = set(range(43, 69))
        number_of_trouble_cyrillic_letters = 0
        number_of_not_trouble_cyrillic_letters = 0
        number_of_trouble_latin_letters = 0
        number_of_not_trouble_latin_letters = 0
        answer = []
        letters = Input.get_letters(path)
        for letter in letters:
            prediction = self.predict(letter)
            if self.__letter_processing_needed:
                if letter not in trouble_indexes:
                    if letter in cyrillic_index:
                        number_of_not_trouble_cyrillic_letters += 1
                    else:
                        number_of_not_trouble_latin_letters += 1
                else:
                    if letter in cyrillic_index:
                        number_of_trouble_cyrillic_letters += 1
                    else:
                        number_of_trouble_latin_letters += 1
            answer.append(prediction)
        expected_language_is_cyrillic = determine_the_expected_language_is_cyrillic(number_of_trouble_cyrillic_letters, number_of_not_trouble_cyrillic_letters, number_of_trouble_latin_letters, number_of_not_trouble_latin_letters)
        for i in range(len(answer)):
            if self.__letter_processing_needed:
                if expected_language_is_cyrillic:
                    if answer[i] not in cyrillic_index:
                        answer[i] = trouble_pairs[answer[i]]
                else:
                    if answer[i] not in latin_index:
                        answer[i] = trouble_pairs[answer[i]]
            answer[i] = alph[answer[i]]

        return ''.join(answer)
