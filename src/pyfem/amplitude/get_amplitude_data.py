# -*- coding: utf-8 -*-
"""

"""
from pyfem.amplitude.BaseAmplitude import BaseAmplitude
from pyfem.amplitude.TabularAmplitude import TabularAmplitude
from pyfem.io.Amplitude import Amplitude
from pyfem.utils.colors import error_style

amplitude_data_dict = {
    'TabularAmplitude': TabularAmplitude
}


def get_amplitude_data(amplitude: Amplitude) -> BaseAmplitude:
    class_name = f'{amplitude.type}'.strip().replace(' ', '')

    if class_name in amplitude_data_dict:
        return amplitude_data_dict[class_name](amplitude=amplitude)
    else:
        error_msg = f'{class_name} bc is not supported.\n'
        error_msg += f'The allowed bc types are {list(amplitude_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
