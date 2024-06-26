import os.path

import numpy as np
from tensorflow.keras.models import load_model
from input import Input


class Cataract:

    def __init__(self, model_language_mode: str = 'both_languages') -> None:
        '''

        Parameters
        ----------
        model_language_mode: ['both_languages' | 'cyrillic' |  'latin' | digits]
        '''
        languages = {
            'both_languages': 'both_languages_model.h5',
            'cyrillic': 'cyrillic_language_model.h5',
            'latin': 'latin_language_model.h5',
            'digits': 'digits_model.h5'
        }
        if model_language_mode in languages.keys():
            if model_language_mode == 'both_languages':
                self.__letter_processing_required = True
            else:
                self.__letter_processing_required = False
            self._model = load_model(os.path.join('models/', languages[model_language_mode]))
        else:
            raise ValueError('incorrect language mode')

    def __str__(self):

        message = ('''                    ----------------
                        Cataract
                     Version: 1.0
                     Supported languages:
                     - Russian
                     - English
                     - Numbers
                     ----------------    
                   ''')
        return message

    def _predict(self, input: np.ndarray) -> int:
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

        def process_letters(list_of_letter_indexes: list[int]) -> list[int]:
            '''
            Parameters
            ----------
            list_of_letter_indexes

            Returns
            -------
            list_of_letter_indexes
            '''
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
            trouble_indexes = [10, 12, 14, 15, 21, 23, 24, 25, 27, 28, 29, 30, 32, 43, 44, 45, 46, 47, 50, 53, 55, 57,
                               58, 62, 66, 67]
            cyrillic_indexes = set(range(10, 43))
            latin_indexes = set(range(43, 69))
            number_of_trouble_cyrillic_letters = 0
            number_of_not_trouble_cyrillic_letters = 0
            number_of_trouble_latin_letters = 0
            number_of_not_trouble_latin_letters = 0
            for index in list_of_letter_indexes:
                if index in trouble_indexes:
                    if index in cyrillic_indexes:
                        number_of_trouble_cyrillic_letters += 1
                    elif index in latin_indexes:
                        number_of_trouble_latin_letters += 1
                else:
                    if index in cyrillic_indexes:
                        number_of_not_trouble_cyrillic_letters += 1
                    elif index in latin_indexes:
                        number_of_not_trouble_latin_letters += 1
            max_not_trouble_letters = max(number_of_not_trouble_latin_letters, number_of_not_trouble_cyrillic_letters)
            if max_not_trouble_letters != 0:
                if number_of_not_trouble_cyrillic_letters >= number_of_not_trouble_latin_letters:
                    expected_language_is_cyrillic = True
                else:
                    expected_language_is_cyrillic = False
            else:
                if number_of_trouble_cyrillic_letters >= number_of_trouble_latin_letters:
                    expected_language_is_cyrillic = True
                else:
                    expected_language_is_cyrillic = False
            for i in range(len(list_of_letter_indexes)):
                if expected_language_is_cyrillic:
                    if list_of_letter_indexes[i] not in cyrillic_indexes and list_of_letter_indexes[i] in trouble_indexes:
                        list_of_letter_indexes[i] = trouble_pairs[list_of_letter_indexes[i]]
                else:
                    if list_of_letter_indexes[i] not in latin_indexes and list_of_letter_indexes[i] in trouble_indexes:
                        list_of_letter_indexes[i] = trouble_pairs[list_of_letter_indexes[i]]
            return list_of_letter_indexes

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

        answer = ''
        letter_indexes = []
        letters = Input.get_letters(path)
        for letter in letters:
            prediction = self._predict(letter)
            letter_indexes.append(prediction)
        if self.__letter_processing_required:
            process_letters(letter_indexes)
        for index in letter_indexes:
            answer += alph[index]
        return answer



