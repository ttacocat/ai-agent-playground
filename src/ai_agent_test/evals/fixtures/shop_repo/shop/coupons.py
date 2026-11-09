from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Coupon:
    code: str
    discount: Decimal  # 与 pricing.apply_discount 相同：0.1 表示减 10%
    expires: date


_COUPONS = {
    c.code: c
    for c in [
        Coupon("SAVE10", Decimal("0.1"), date(2099, 12, 31)),
        Coupon("OLD20", Decimal("0.2"), date(2020, 1, 1)),
    ]
}


def find(code: str) -> Coupon | None:
    """按券码查询优惠券，不区分大小写。不存在时返回 None。"""
    return _COUPONS.get(code.strip().upper())
