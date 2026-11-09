from decimal import Decimal

from shop.pricing import apply_discount, subtotal


def test_apply_discount():
    assert apply_discount(Decimal("100"), Decimal("0.2")) == Decimal("80.00")


def test_subtotal():
    assert subtotal([(Decimal("9.99"), 2), (Decimal("0.01"), 1)]) == Decimal("19.99")
