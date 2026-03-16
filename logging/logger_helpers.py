import logging

from logger import SimulationLogger


def setup_logger():
    logger = logging.getLogger("HANDOVER_ENV")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(SimulationLogger())
        logger.addHandler(handler)
    return logger


def log_event(step: int, ue_id: int, event: str, **kwargs):
    details = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger = logging.getLogger("HANDOVER_ENV")
    logger.info(
        "t=%04d | UE=%02d | EVENT=%s | %s",
        step,
        ue_id,
        event,
        details,
    )


def log_error(step: int, event: str, **kwargs):
    details = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger = logging.getLogger("HANDOVER_ENV")
    logger.error(
        "t=%04d | EVENT=%S | %s",
        step,
        event,
        details,
    )
