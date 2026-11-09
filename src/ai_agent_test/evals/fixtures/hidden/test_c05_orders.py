from decimal import Decimal

import pytest
from shop.errors import StockError
from shop.orders import can_fulfill, place_order


def test_boundaries():
    assert can_fulfill("kb-01", 5)
    assert can_fulfill("kb-01", 4)
    assert not can_fulfill("kb-01", 6)
    assert not can_fulfill("ms-02", 1)


def test_place_order_reserves_stock():
    assert place_order("kb-01", 5, Decimal("2")) == Decimal("10.00")
    with pytest.raises(StockError):
        place_order("kb-01", 1, Decimal("2"))
