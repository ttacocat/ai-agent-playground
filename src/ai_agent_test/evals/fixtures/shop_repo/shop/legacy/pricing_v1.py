"""旧版计价（v1）。已废弃：只为兼容旧报表保留，新代码不要使用。"""

COUPON_PERCENT = {"SAVE10": 10, "OLD20": 20, "VIP30": 30}


def apply_coupon(price: float, code: str) -> float:
    """按优惠券打折（旧版：百分比、float、不检查有效期）。"""
    percent = COUPON_PERCENT.get(code, 0)
    return round(price * (100 - percent) / 100, 2)
