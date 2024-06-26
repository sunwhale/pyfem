# -*- coding: utf-8 -*-
"""

"""
from typing import Callable

from numpy import array, all, diff, ndarray
from scipy.interpolate import interp1d  # type: ignore

from pyfem.amplitude.BaseAmplitude import BaseAmplitude
from pyfem.io.Amplitude import Amplitude
from pyfem.utils.colors import error_style


class TabularAmplitude(BaseAmplitude):
    """
    通过x-y数据表格定义的幅值。

    :ivar table: x-y数据表格
    :vartype table: ndarray
    """

    __slots_dict__: dict = {
        'table': ('ndarray', 'x-y数据表格')
    }

    __slots__ = BaseAmplitude.__slots__ + [slot for slot in __slots_dict__.keys()]

    def __init__(self, amplitude: Amplitude) -> None:
        super().__init__()
        self.start = amplitude.start
        self.table: ndarray = array(amplitude.data)
        if self.table.ndim != 2:
            raise NotImplementedError(error_style('dimension of amplitude table must be 2'))
        elif self.table.shape[1] != 2:
            raise NotImplementedError(error_style('column of amplitude table must be 2'))
        elif not all(diff(self.table[:, 0]) > 0):
            raise NotImplementedError(error_style('time of amplitude table must be monotonic increase'))
        self.f_amplitude: Callable = interp1d(self.table[:, 0], self.table[:, 1], kind='linear',
                                              fill_value='extrapolate')

    def get_amplitude(self, time: float) -> float:
        return self.f_amplitude(time) + self.start


if __name__ == "__main__":
    from pyfem.utils.visualization import print_slots_dict

    print_slots_dict(TabularAmplitude.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    amp = TabularAmplitude(props.amplitudes[0])
    amp.show()
