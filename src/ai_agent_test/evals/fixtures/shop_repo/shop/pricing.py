from decimal import Decimal

from shop.utils import round_money


def apply_discount(price: Decimal, discount: Decimal) -> Decimal:
    """按折扣率打折。discount 是 0 到 1 之间的小数，0.2 表示减价 20%（打八折）。"""
    return round_money(price * (1 - discount))


def subtotal(items: list[tuple[Decimal, int]]) -> Decimal:
    """items 为 (单价, 数量) 的列表。"""
    return round_money(sum((price * qty for price, qty in items), Decimal("0")))
