class ShopError(Exception):
    """商店业务错误的基类。"""


class PricingError(ShopError):
    """计价相关的错误。"""


class StockError(ShopError):
    """库存相关的错误。"""
