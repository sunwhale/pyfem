# -*- coding: utf-8 -*-
"""

"""
from typing import Union

from pyfem.amplitude.BaseAmplitude import BaseAmplitude
from pyfem.amplitude.TabularAmplitude import TabularAmplitude
from pyfem.io.Amplitude import Amplitude
from pyfem.utils.colors import error_style

AmplitudeData = Union[BaseAmplitude, TabularAmplitude]

amplitude_data_dict = {
    'TabularAmplitude': TabularAmplitude
}


def get_amplitude_data(amplitude: Amplitude) -> AmplitudeData:
    """
    工厂函数，用于根据幅值属性生产不同的幅值对象。

    Args:
        amplitude(Amplitude): 幅值属性

    :return: 幅值对象
    :rtype: AmplitudeData
    """

    class_name = f'{amplitude.type}'.strip().replace(' ', '')

    if class_name in amplitude_data_dict:
        return amplitude_data_dict[class_name](amplitude=amplitude)
    else:
        error_msg = f'{class_name} bc is not supported.\n'
        error_msg += f'The allowed bc types are {list(amplitude_data_dict.keys())}.'
        raise NotImplementedError(error_style(error_msg))


if __name__ == "__main__":
    pass
