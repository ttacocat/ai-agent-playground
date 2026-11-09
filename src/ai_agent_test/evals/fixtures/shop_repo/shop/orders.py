from decimal import Decimal

from shop.errors import StockError
from shop.inventory import get_stock, reserve
from shop.pricing import subtotal


def can_fulfill(sku: str, qty: int) -> bool:
    """库存是否足够满足这次下单。"""
    return get_stock(sku) >= qty


def place_order(sku: str, qty: int, price: Decimal) -> Decimal:
    """下单：检查并扣减库存，返回应付金额。"""
    if not can_fulfill(sku, qty):
        raise StockError(f"{sku} 库存不足")
    reserve(sku, qty)
    return subtotal([(price, qty)])
