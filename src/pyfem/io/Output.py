# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple

from pyfem.io.BaseIO import BaseIO


class Output(BaseIO):
    __slots__: Tuple = ('type',
                        'field_outputs',
                        'on_screen')

    def __init__(self) -> None:
        super().__init__()
        self.type: str = None  # type: ignore
        self.field_outputs: str = None  # type: ignore
        self.on_screen: bool = None  # type: ignore


if __name__ == "__main__":
    output = Output()
    output.show()
