import logging
import os
import sys
import time
import unicodedata

project_name = os.getcwd().split('/')[-1]
_logger = logging.getLogger(project_name)
_logger.propagate = False
_logger.addHandler(logging.StreamHandler())


def _log_prefix():
    def _get_file_line():
        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        f = sys._getframe()
        # pylint: enable=protected-access
        our_file = f.f_code.co_filename
        f = f.f_back
        while f:
            code = f.f_code
            if code.co_filename != our_file:
                return code.co_filename, f.f_lineno
            f = f.f_back
        return '<unknown>', 0

    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_millisecond = int(1e3 * (now % 1.0))

    # current filename and line
    filename, line = _get_file_line()
    basename = os.path.basename(filename)

    s = '%02d-%02d %02d:%02d:%02d.%03d %s:%d] ' % (
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_millisecond,
        basename,
        line)

    return s

def logging_verbosity(verbosity=0):
    _logger.setLevel(verbosity)

def add_filehandler(filepath):
    _logger.addHandler(logging.FileHandler(filepath, mode="a"))

def debug(msg, *args, **kwargs):
    _logger.debug('D ' + project_name + ' ' + _log_prefix() + msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    _logger.info(_log_prefix() + msg, *args, **kwargs)

def warn(msg, *args, **kwargs):
    _logger.warning('W ' + project_name + ' ' + _log_prefix() + msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    _logger.error('E ' + project_name + ' ' + _log_prefix() + msg, *args, **kwargs)

def fatal(msg, *args, **kwargs):
    _logger.fatal('F ' + project_name + ' ' + _log_prefix() + msg, *args, **kwargs)

def preformat_cjk(string, width, align='<', fill=' '):
    count = (width - sum(1 + (unicodedata.east_asian_width(c) in "WF")
                         for c in string))
    return {
        '>': lambda s: fill * count + s,
        '<': lambda s: s + fill * count,
        '^': lambda s: fill * (count / 2)
                       + s
                       + fill * (count / 2 + count % 2)
    }[align](string)
