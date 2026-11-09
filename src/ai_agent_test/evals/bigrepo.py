"""大仓库：在示例仓库上确定性地生成几十个模块，用来测"需要广泛搜索"的任务。

小仓库只有十几个文件，单 agent 全读一遍毫无压力，测不出子 agent 的价值。
这里加上：

- 24 个商品类目模块（``shop/catalog/*.py``），其中 6 个违反约定、用内置 ``round()`` 算会员价；
- 一组看起来相关、其实无关的服务模块（名字里带 shipping 的就有好几个）；
- 运费规则藏在 ``shop/fulfillment/fees.py``，满额免运费的边界写错了；
- 4 处调用 ``coupons.find``，写法各不相同（直接调用、``import ... as``、``from ... import find as``），
  搜 "coupons.find" 会漏掉一部分。

文件在运行时生成，不落盘：仓库内容由代码决定，每次都一样，也不用维护上百个样板文件。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from functools import cache

CATEGORIES = [
    "books", "toys", "garden", "kitchen", "sports", "music", "office", "pets",
    "beauty", "baby", "auto", "tools", "games", "shoes", "bags", "watches",
    "lights", "phones", "cameras", "audio", "drinks", "snacks", "tea", "coffee",
]  # fmt: skip
# 用内置 round(float) 算钱、违反约定的类目
ROUND_VIOLATORS = ("garden", "pets", "watches", "audio", "snacks", "tea")
MEMBER_RATE = Decimal("0.95")

SERVICES = [
    "notifications", "search_index", "recommendations", "reviews", "wishlist", "returns",
    "loyalty", "gift_cards", "invoices", "payments", "fraud_check", "analytics",
    "shipping_estimate", "shipping_labels", "address_book", "customer_support", "newsletter",
    "price_alerts", "store_locator", "warehouse_sync", "supplier_feed", "tax_reports",
    "checkout_preview", "session_store", "feature_flags", "audit_export",
]  # fmt: skip

COUPON_CALLERS = {
    "shop/marketing/campaigns.py": "direct",
    "shop/services/checkout_preview.py": "module_alias",
    "shop/reports/coupon_usage.py": "function_alias",
    "scripts/expire_coupons.py": "direct",
}


@cache
def tricky_bases() -> tuple[Decimal, ...]:
    """float 的 round 和 ROUND_HALF_UP 结果不同的价格：用来抓"还在用 round()"的实现。"""
    found = []
    for cents in range(1, 3000):
        base = Decimal(cents) / 100
        half_up = (base * MEMBER_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        builtin = Decimal(str(round(float(base) * float(MEMBER_RATE), 2)))
        if half_up != builtin:
            found.append(base)
        if len(found) == 5:
            break
    return tuple(found)


def _catalog(category: str) -> str:
    title = category.replace("_", " ")
    if category in ROUND_VIOLATORS:
        member = (
            "def member_price(base: Decimal) -> Decimal:\n"
            '    """会员价：九五折。"""\n'
            "    return Decimal(str(round(float(base) * 0.95, 2)))\n"
        )
    else:
        member = (
            "def member_price(base: Decimal) -> Decimal:\n"
            '    """会员价：九五折。"""\n'
            "    return round_money(base * MEMBER_RATE)\n"
        )
    return (
        f'"""{title} 类目：商品展示与价格规则。"""\n\n'
        "from decimal import Decimal\n\n"
        "from shop.utils import round_money\n\n"
        'MEMBER_RATE = Decimal("0.95")\n'
        f'CATEGORY = "{category}"\n\n\n'
        f"{member}\n\n"
        "def bundle_price(prices: list[Decimal]) -> Decimal:\n"
        '    """三件及以上打九折。"""\n'
        '    total = sum(prices, Decimal("0"))\n'
        "    if len(prices) >= 3:\n"
        '        total = total * Decimal("0.9")\n'
        "    return round_money(total)\n\n\n"
        "def display_name(sku: str) -> str:\n"
        f'    return f"[{title}] {{sku}}"\n'
    )


def _service(name: str) -> str:
    title = name.replace("_", " ")
    body = (
        f'"""{title}：与订单主流程无关的周边服务。"""\n\n'
        "from shop.log import get_logger\n\n"
        "logger = get_logger(__name__)\n\n\n"
        f"def {name}_enabled() -> bool:\n"
        "    return True\n\n\n"
        f"def run_{name}(payload: dict) -> dict:\n"
        f'    """执行 {title}。"""\n'
        f'    logger.info("{name}: %s", sorted(payload))\n'
        '    return {"ok": True, "service": "' + name + '"}\n'
    )
    if name == "shipping_estimate":
        body += (
            "\n\n"
            "def estimate_days(zone: str) -> int:\n"
            '    """预计送达天数（只估时效，不算运费）。"""\n'
            '    return {"local": 1, "domestic": 3}.get(zone, 7)\n'
        )
    if name == "checkout_preview":
        body += (
            "\n\n"
            "import shop.coupons as cp  # noqa: E402\n\n\n"
            "def preview_coupon(code: str) -> str:\n"
            "    coupon = cp.find(code)\n"
            '    return "无效" if coupon is None else f"减 {coupon.discount}"\n'
        )
    return body


FILES_EXTRA = {
    "shop/catalog/__init__.py": "",
    "shop/services/__init__.py": "",
    "shop/marketing/__init__.py": "",
    "shop/fulfillment/__init__.py": "",
    "shop/fulfillment/fees.py": '''"""运费规则。"""

from decimal import Decimal

from shop.utils import round_money

FREE_SHIPPING_THRESHOLD = Decimal("99")
BASE_FEE = {"standard": Decimal("8"), "express": Decimal("20")}


def shipping_fee(subtotal: Decimal, method: str = "standard") -> Decimal:
    """标准配送满 FREE_SHIPPING_THRESHOLD 元（含）免运费；加急配送不免。"""
    if method == "standard" and subtotal > FREE_SHIPPING_THRESHOLD:
        return Decimal("0.00")
    return round_money(BASE_FEE[method])
''',
    "shop/fulfillment/carriers.py": '''"""承运商。"""

CARRIERS = {"standard": "邮政", "express": "顺丰"}


def carrier_for(method: str) -> str:
    return CARRIERS.get(method, "邮政")
''',
    "shop/fulfillment/packaging.py": '''"""包装规格。"""


def box_size(items: int) -> str:
    return "S" if items <= 2 else "M" if items <= 6 else "L"
''',
    "shop/marketing/campaigns.py": '''"""营销活动：检查活动页上展示的优惠券是否还能用。"""

from shop import coupons


def campaign_coupon_valid(code: str) -> bool:
    return coupons.find(code) is not None
''',
    "shop/reports/coupon_usage.py": '''"""优惠券使用报表。"""

from shop.coupons import find as find_coupon


def describe(codes: list[str]) -> list[str]:
    return [code if find_coupon(code) else f"{code}（已失效）" for code in codes]
''',
    "scripts/expire_coupons.py": '''"""运维脚本：列出已经查不到的优惠券。

用法：python scripts/expire_coupons.py CODE...
"""

import sys

from shop import coupons

if __name__ == "__main__":
    for code in sys.argv[1:]:
        if coupons.find(code) is None:
            print(code)
''',
}


@cache
def big_repo_files() -> dict[str, str]:
    files = dict(FILES_EXTRA)
    for category in CATEGORIES:
        files[f"shop/catalog/{category}.py"] = _catalog(category)
    for name in SERVICES:
        files[f"shop/services/{name}.py"] = _service(name)
    return files


def hidden_catalog_test() -> str:
    """C10 的隐藏测试：所有类目的会员价都要等于 round_money(base × 0.95)，且是 Decimal。"""
    bases = ", ".join(f'Decimal("{b}")' for b in tricky_bases())
    modules = ", ".join(f'"{c}"' for c in CATEGORIES)
    return f"""import importlib
from decimal import Decimal

import pytest

from shop.utils import round_money


@pytest.mark.parametrize("category", [{modules}])
@pytest.mark.parametrize("base", [{bases}])
def test_member_price_uses_round_money(category, base):
    module = importlib.import_module(f"shop.catalog.{{category}}")
    result = module.member_price(base)
    assert isinstance(result, Decimal)
    assert result == round_money(base * Decimal("0.95"))
"""


HIDDEN_SHIPPING_TEST = """from decimal import Decimal

from shop.fulfillment.fees import shipping_fee


def test_free_shipping_includes_threshold():
    assert shipping_fee(Decimal("99")) == Decimal("0.00")
    assert shipping_fee(Decimal("150")) == Decimal("0.00")


def test_below_threshold_pays():
    assert shipping_fee(Decimal("98.99")) == Decimal("8.00")


def test_express_never_free():
    assert shipping_fee(Decimal("99"), "express") == Decimal("20.00")
"""
