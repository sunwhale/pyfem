import os

IS_COLORED = True

if os.environ.get('TERM', '') != '' or IS_COLORED:
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
else:
    CYAN = ''
    MAGENTA = ''
    BLUE = ''
    GREEN = ''
    YELLOW = ''
    RED = ''
    BOLD = ''
    UNDERLINE = ''
    END = ''


def error_style(error_msg: str) -> str:
    return MAGENTA + BOLD + error_msg + END


def insert_spaces(n: int, text: str) -> str:
    lines = text.split('\n')
    indented_lines = [' ' * n + line for line in lines]
    return '\n'.join(indented_lines)
