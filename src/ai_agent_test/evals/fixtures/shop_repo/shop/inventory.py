from shop import audit
from shop.errors import StockError

_STOCK: dict[str, int] = {"kb-01": 5, "ms-02": 0, "hd-03": 12}


def get_stock(sku: str) -> int:
    """查询库存数量。商品不存在时抛 StockError。"""
    if sku not in _STOCK:
        raise StockError(f"未知商品：{sku}")
    return _STOCK[sku]


def reserve(sku: str, qty: int) -> None:
    """扣减库存。库存不足时抛 StockError。"""
    if get_stock(sku) < qty:
        raise StockError(f"{sku} 库存不足")
    _STOCK[sku] -= qty
    audit.record("reserve", sku, -qty)
