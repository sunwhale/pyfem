import logging
from typing import Optional

import colorlog


def set_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> Optional[logging.Logger]:
    """
    设置 logging 模块的基础配置，并返回一个 logger 对象。
    :param log_file: 日志文件名，None 表示将日志输出到控制台
    :param level: 输出日志的最低级别，默认是 INFO 级别
    """
    # 创建 logger 对象
    logger = logging.getLogger()

    # 设置日志级别
    logger.setLevel(level)

    # 创建格式化器
    if log_file:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = colorlog.ColoredFormatter('%(log_color)s%(message)s')

    # 创建处理器
    if log_file:
        # 日志输出到文件
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # 日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger():
    return logging.getLogger()


if __name__ == "__main__":
    set_logger()
    logger = get_logger()
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")
