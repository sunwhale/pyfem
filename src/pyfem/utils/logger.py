# -*- coding: utf-8 -*-
"""

"""
import logging
from logging import Logger
from typing import Optional

import colorlog

logger = logging.getLogger()


def set_logger(logger: Logger, job, level: int = logging.DEBUG) -> Optional[logging.Logger]:
    """
    设置 logging 模块的基础配置，并返回一个 logger 对象。

    :param job: Job对象
    :param level: 输出日志的最低级别，默认是 DEBUG 级别
    """
    # 创建 logger 对象

    # 设置日志级别
    logger.setLevel(level)

    # 创建处理器
    log_file = job.abs_input_file.with_suffix('.log')
    log_file_handler = logging.FileHandler(log_file, mode='w')
    log_file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)

    sta_file = job.abs_input_file.with_suffix('.sta')
    sta_file_handler = logging.FileHandler(sta_file, mode='w')
    sta_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    sta_file_handler.setFormatter(formatter)
    logger.addHandler(sta_file_handler)

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
