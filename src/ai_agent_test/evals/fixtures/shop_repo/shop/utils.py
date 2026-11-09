from decimal import ROUND_HALF_UP, Decimal


def round_money(value: Decimal) -> Decimal:
    """金额保留两位小数，四舍五入（ROUND_HALF_UP）。"""
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def parse_quantity(text: str) -> int:
    """把用户输入的数量解析成正整数。

    允许前后有空格，允许带"件"字，例如 " 3 "、"3件"。不是正整数时抛 ValueError。
    """
    cleaned = text.strip().removesuffix("件").strip()
    if not cleaned.isdigit():
        raise ValueError(f"无效的数量：{text!r}")
    value = int(cleaned)
    if value <= 0:
        raise ValueError(f"数量必须大于 0：{text!r}")
    return value
