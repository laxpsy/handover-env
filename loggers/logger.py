import logging


class LoggerColors:
    grey = "\x1b[0;37m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[1;31m"
    purple = "\x1b[1;35m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    reset = "\x1b[0m"
    blink_red = "\x1b[5m\x1b[1;31m"


class SimulationLogger(logging.Formatter):
    currTime = f"{LoggerColors.grey}[%(asctime)s]{LoggerColors.reset}"
    base_format = "[%(levelname)s] | [%(filename)s:%(lineno)d] | %(message)s"

    FORMATS = {
        logging.DEBUG: currTime
        + " | "
        + LoggerColors.purple
        + base_format
        + LoggerColors.reset,
        logging.INFO: currTime
        + " | "
        + LoggerColors.light_blue
        + base_format
        + LoggerColors.reset,
        logging.ERROR: currTime
        + " | "
        + LoggerColors.red
        + base_format
        + LoggerColors.reset,
        logging.WARNING: currTime
        + " | "
        + LoggerColors.yellow
        + base_format
        + LoggerColors.reset,
        logging.CRITICAL: currTime
        + " | "
        + LoggerColors.blink_red
        + base_format
        + LoggerColors.reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return super().format(record) if log_fmt is None else formatter.format(record)
