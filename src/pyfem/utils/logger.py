# -*- coding: utf-8 -*-
"""

"""
import logging
from logging import Logger
from typing import Optional

import colorlog

logger = logging.getLogger()


def set_logger(logger: Logger, log_file: Optional[str] = None, level: int = logging.DEBUG) -> Optional[logging.Logger]:
    """
    设置 logging 模块的基础配置，并返回一个 logger 对象。

    :param log_file: 日志文件名，None 表示将日志输出到控制台
    :param level: 输出日志的最低级别，默认是 DEBUG 级别
    """
    # 创建 logger 对象

    # 设置日志级别
    logger.setLevel(level)

    # 创建处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = colorlog.ColoredFormatter('%(log_color)s%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    set_logger(logger, level=logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
