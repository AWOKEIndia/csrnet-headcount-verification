import os
import logging
from datetime import datetime
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO, console=True):
    """
    Set up a logger with both file and console handlers.

    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to the log file. If None, logs will only be printed to console
        level (int, optional): Logging level. Defaults to logging.INFO
        console (bool, optional): Whether to print logs to console. Defaults to True

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Add file handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Add timestamp to log filename if it doesn't exist
        if not os.path.exists(log_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base, ext = os.path.splitext(log_file)
            log_file = f"{base}_{timestamp}{ext}"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Add console handler if console is True
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

def get_logger(name, log_dir='logs', level=logging.INFO):
    """
    Convenience function to get a logger with default settings.

    Args:
        name (str): Name of the logger
        log_dir (str, optional): Directory to store log files. Defaults to 'logs'
        level (int, optional): Logging level. Defaults to logging.INFO

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(log_dir, f'{name}.log')

    return setup_logger(name, log_file, level)

def log_system_info(logger):
    """
    Log system information including Python version, OS, and hardware details.

    Args:
        logger (logging.Logger): Logger instance to use
    """
    import platform
    import psutil
    import torch

    logger.info("System Information:")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) Available: Yes")
    else:
        logger.info("CUDA/MPS Available: No")

def log_exception(logger, e, context=None):
    """
    Log an exception with optional context information.

    Args:
        logger (logging.Logger): Logger instance to use
        e (Exception): Exception to log
        context (dict, optional): Additional context information
    """
    logger.error(f"Exception occurred: {str(e)}")
    if context:
        logger.error(f"Context: {context}")
    logger.error("Stack trace:", exc_info=True)

def log_metrics(logger, metrics, prefix=""):
    """
    Log metrics in a formatted way.

    Args:
        logger (logging.Logger): Logger instance to use
        metrics (dict): Dictionary of metrics to log
        prefix (str, optional): Prefix to add to metric names
    """
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{prefix}{name}: {value:.4f}")
        else:
            logger.info(f"{prefix}{name}: {value}")
