from shop.inventory import _STOCK, get_stock


def low_stock(threshold: int = 3) -> list[str]:
    """库存低于阈值的商品编号，按编号排序。"""
    return sorted(sku for sku in _STOCK if get_stock(sku) < threshold)
