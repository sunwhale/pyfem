# -*- coding: utf-8 -*-
"""

"""
import logging
import time
from logging import Logger
from pathlib import Path
from typing import Optional

import colorlog

from pyfem import __version__

logging.addLevelName(21, 'IN21')
logging.addLevelName(22, 'IN22')

logger = logging.getLogger('log')
logger_sta = logging.getLogger('sta')

STA_HEADER = f"""PYFEM {__version__} DATE {time.strftime('%Y-%m-%d', time.localtime())} TIME {time.strftime('%H:%M:%S', time.localtime())}
SUMMARY OF JOB INFORMATION:
STEP  INCREMENT  ATT  SEVERE  EQUIL  TOTAL      TOTAL TIME       STEP TIME     INC OF TIME  DOF      IF
                      DISCON  ITERS  ITERS                                                  MONITOR  RIKS
                      ITERS"""


def set_logger(logger: Logger, abs_input_file: Path, level: int = logging.DEBUG) -> Optional[logging.Logger]:
    """
    设置 logging 模块的基础配置，并返回一个 logger 对象。

    :param logger: 要配置的logger对象
    :param abs_input_file: Job配置文件的绝对路径
    :param level: 输出日志的最低级别，默认是 DEBUG 级别
    """
    # 创建 logger 对象

    # 设置日志级别
    logger.setLevel(level)

    # 创建处理器
    log_file = abs_input_file.with_suffix('.log')
    log_file_handler = logging.FileHandler(log_file, mode='w')
    log_file_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = colorlog.ColoredFormatter('%(log_color)s%(message)s',
                                          log_colors={
                                              'DEBUG': 'white',
                                              'INFO': 'green',
                                              'WARNING': 'yellow',
                                              'ERROR': 'red',
                                              'CRITICAL': 'red,bg_white',
                                              'IN21': 'white',
                                              'IN22': 'purple',
                                          })

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def set_logger_sta(logger: Logger, abs_input_file: Path, level: int = logging.INFO) -> Optional[logging.Logger]:
    """
    设置 logging 模块的基础配置，并返回一个 logger 对象。

    :param logger: 要配置的logger对象
    :param abs_input_file: Job配置文件的绝对路径
    :param level: 输出日志的最低级别，默认是 DEBUG 级别
    """
    # 创建 logger 对象

    # 设置日志级别
    logger.setLevel(level)

    # 创建处理器
    sta_file = abs_input_file.with_suffix('.sta')
    sta_file_handler = logging.FileHandler(sta_file, mode='w')
    sta_file_handler.setLevel(level)
    formatter = logging.Formatter('%(message)s')
    sta_file_handler.setFormatter(formatter)
    logger.addHandler(sta_file_handler)

    return logger


if __name__ == "__main__":
    print(STA_HEADER)
