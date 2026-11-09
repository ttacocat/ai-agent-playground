from decimal import Decimal

# 各地区的税率。新增地区只改这里。
TAX_RATES: dict[str, Decimal] = {
    "CN": Decimal("0.13"),
    "HK": Decimal("0"),
    "US": Decimal("0.0725"),
}
