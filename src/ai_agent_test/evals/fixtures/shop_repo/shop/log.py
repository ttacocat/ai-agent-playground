import logging


def get_logger(name: str) -> logging.Logger:
    """项目统一的 logger，名字以 shop. 开头，方便统一配置。"""
    return logging.getLogger(name if name.startswith("shop") else f"shop.{name}")
