from decimal import Decimal

import pytest
from shop import tax
from shop.errors import PricingError
from shop.pricing import total_with_tax


def test_total_with_tax_basic():
    assert total_with_tax([(Decimal("100"), 2)], "CN") == Decimal("226.00")
    assert total_with_tax([(Decimal("100"), 1)], "HK") == Decimal("100.00")


def test_rounds_half_up_like_round_money():
    # 0.50 × 1.13 = 0.565：ROUND_HALF_UP 得 0.57；内置 round() 或默认的 HALF_EVEN 得 0.56
    assert total_with_tax([(Decimal("0.50"), 1)], "CN") == Decimal("0.57")


def test_reads_rates_from_tax_table(monkeypatch):
    # README 约定：税率只在 TAX_RATES 里维护。写死 0.13 的实现在这里会失败
    monkeypatch.setitem(tax.TAX_RATES, "CN", Decimal("0.5"))
    assert total_with_tax([(Decimal("100"), 1)], "CN") == Decimal("150.00")


def test_unknown_region_raises_pricing_error():
    with pytest.raises(PricingError):
        total_with_tax([(Decimal("1"), 1)], "XX")
