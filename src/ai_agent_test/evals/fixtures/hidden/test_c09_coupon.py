from decimal import Decimal

import pytest
from shop.errors import PricingError
from shop.inventory import get_stock
from shop.orders import place_order


def test_coupon_applies_to_subtotal():
    result = place_order("hd-03", 2, Decimal("10.00"), coupon="SAVE10")
    # shop/legacy 里的旧实现返回 float；新代码金额必须是 Decimal
    assert isinstance(result, Decimal)
    assert result == Decimal("18.00")


def test_without_coupon_unchanged():
    assert place_order("hd-03", 1, Decimal("10.00")) == Decimal("10.00")


@pytest.mark.parametrize("code", ["OLD20", "NOPE"])
def test_expired_or_unknown_coupon_raises(code):
    # OLD20 已过期：旧实现不检查有效期
    with pytest.raises(PricingError):
        place_order("hd-03", 1, Decimal("10.00"), coupon=code)


def test_rejected_coupon_does_not_reserve_stock():
    # docs/CONVENTIONS.md：下单失败时不能扣减库存
    before = get_stock("hd-03")
    with pytest.raises(PricingError):
        place_order("hd-03", 1, Decimal("10.00"), coupon="OLD20")
    assert get_stock("hd-03") == before
