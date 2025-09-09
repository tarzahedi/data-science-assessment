import logging


def get_run_logger() -> logging.Logger:
    """Create and configure a logger for runtime messages.
    
    (Will use prefect logger instead of this)

    Returns:
        logging.Logger: A configured logger instance for emitting runtime logs.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # show DEBUG and above
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger
