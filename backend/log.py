import logging


_Logger = logging.getLogger('_log_.event')


def info(msg):
    _Logger.info(msg)


def debug(msg):
    _Logger.debug(msg)


def error(msg):
    _Logger.error(msg)
