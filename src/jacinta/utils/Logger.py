from __future__ import annotations

import logging
from datetime import datetime


class Logger:

    class _CustomFormatter(logging.Formatter):

        def __init__(self) -> None:
            """
            Custom log formatter.

            Args:
                None

            Returns:
                None
            """
            fmt = "[%(asctime)s] [%(name)s.%(funcName)s] [%(levelname)s] %(message)s"
            datefmt = "%Y/%m/%d %H:%M:%S.%f"
            super().__init__(fmt=fmt, datefmt=datefmt)
            return

        def formatTime(
            self, record: logging.LogRecord, datefmt: str | None = None
        ) -> str:
            """
            Format timestamp for log record.

            Args:
                record (logging.LogRecord): Log record being formatted
                datefmt (str | None): Optional explicit datetime format

            Returns:
                str: Formatted timestamp including microseconds
            """
            current = datetime.fromtimestamp(record.created)
            current = (
                current.strftime(datefmt)
                if datefmt
                else current.strftime("%Y/%m/%d %H:%M:%S.%f")
            )
            return current

    def __init__(self, obj: str) -> None:
        """
        Initialize logger wrapper.

        Args:
            obj (str): Logger name, usually __name__ or class name

        Returns:
            None
        """
        self._logger = logging.getLogger(obj)
        self._logger.setLevel(logging.DEBUG)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._CustomFormatter())
            self._logger.addHandler(handler)
        return

    @property
    def logger(self) -> logging.Logger:
        """
        Expose the underlying logger instance.

        Args:
            None

        Returns:
            logging.Logger: Configured logger
        """
        logger = self._logger
        return logger
