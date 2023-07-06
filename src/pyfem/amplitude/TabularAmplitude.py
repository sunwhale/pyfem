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
    pass
