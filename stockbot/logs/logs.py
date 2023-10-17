'''
    Basic Logging Class
        - Debug
        - Info
        - Warning
        - Error
        - Critical
'''
from dataclasses import dataclass
import logging

@dataclass(frozen=True, unsafe_hash=True)
class Logger:

    @staticmethod
    def debug(loggingDict, message, logger):
        logger.debug(Logger.createLogString(loggingDict, message))

    @staticmethod
    def info(loggingDict, message, logger):
        logger.info(Logger.createLogString(loggingDict, message))

    @staticmethod
    def warning(loggingDict, message, logger):
        logger.warning(Logger.createLogString(loggingDict, message))

    @staticmethod
    def error(loggingDict, message, logger):
        logger.error(Logger.createLogString(loggingDict, message))

    @staticmethod
    def critical(loggingDict, message, logger):
        logger.critical(Logger.createLogString(loggingDict, message))

    @staticmethod
    def createLogString(loggingDict, message):
        keyValuesBracket = '[' + ' '.join( [str(k) + '=' + '"' + str(v) + '"'
                                            for k, v in sorted(loggingDict.items())] ) + '] '
        return keyValuesBracket + message

    @staticmethod
    def loggerSetup(formatting="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(formatting)

        # add formatter to ch
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        return logger