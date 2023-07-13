# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, List, Tuple

from pyfem.io.BaseIO import BaseIO


class Dof(BaseIO):
    __slots__: Tuple = ('names',
                        'family',
                        'order')

    def __init__(self) -> None:
        super().__init__()
        self.names: List[str] = None  # type: ignore
        self.family: Optional[str] = None
        self.order: Optional[int] = None


if __name__ == "__main__":
    dof = Dof()
    dof.show()
