import logging
import os


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


class Logger(logging.getLoggerClass()):
    __GREEN = "\033[0;32m%s\033[0m"
    __FORMAT = {
        "fmt": "%(asctime)s %(levelname)s: %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    def __init__(self, format=__FORMAT):
        super().__init__("Logger")
        formatter = logging.Formatter(**format)

        self.root.setLevel(LOG_LEVEL)
        self.root.handlers = []

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.root.addHandler(handler)

    def info(self, message):
        self.root.info(message)

    def info_green(self, message):
        self.root.info(self.__GREEN, message)
