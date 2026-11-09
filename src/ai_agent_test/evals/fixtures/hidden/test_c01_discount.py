from decimal import Decimal

import pytest
from shop.errors import PricingError
from shop.pricing import apply_discount


@pytest.mark.parametrize("discount", ["1.5", "-0.1", "2"])
def test_out_of_range_discount_raises_pricing_error(discount):
    # README 约定：计价问题抛 PricingError，不是 ValueError
    with pytest.raises(PricingError):
        apply_discount(Decimal("100"), Decimal(discount))


@pytest.mark.parametrize(("discount", "expected"), [("0", "100.00"), ("1", "0.00"), ("0.2", "80.00")])
def test_valid_discounts_still_work(discount, expected):
    assert apply_discount(Decimal("100"), Decimal(discount)) == Decimal(expected)
