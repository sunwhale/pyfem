# -*- coding: utf-8 -*-
"""

"""
from typing import List, Tuple

from pyfem.io.BaseIO import BaseIO


class Amplitude(BaseIO):
    __slots__: Tuple = ('name',
                        'type',
                        'start',
                        'data')

    def __init__(self) -> None:
        super().__init__()
        self.name: str = None  # type: ignore
        self.type: str = None  # type: ignore
        self.start: float = None  # type: ignore
        self.data: List = None  # type: ignore


if __name__ == "__main__":
    amp = Amplitude()
    amp.show()
