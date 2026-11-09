from decimal import Decimal

from shop.orders import can_fulfill, place_order


def test_can_fulfill_exact_stock():
    assert can_fulfill("kb-01", 5)


def test_place_order():
    assert place_order("hd-03", 1, Decimal("99.50")) == Decimal("99.50")
