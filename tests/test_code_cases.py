"""编码用例的离线测试。

隐藏测试决定了编码用例的对错，所以它们本身要先被验证：
每条用例都给一份正确解法（必须判通过）和一份"不看项目约定"的踩坑解法（必须判失败）。
"""

from __future__ import annotations

import pytest

from ai_agent_test.evals.cases import CASES, CODE_CASES
from ai_agent_test.evals.stub_tools import build_stub_tools
from ai_agent_test.evals.world import StubWorld, load_repo
from ai_agent_test.guardrails import RiskLevel, assess, read_only

CASE = {c.id: c for c in CASES}
REPO = load_repo()


def world_for(case_id: str) -> StubWorld:
    world = StubWorld()
    if CASE[case_id].setup:
        CASE[case_id].setup(world)
    return world


def patch(world: StubWorld, path: str, old: str, new: str) -> None:
    assert old in world.files[path], f"{path} 里找不到要替换的内容"
    world.files[path] = world.files[path].replace(old, new)


# --------------------------------------------------------------------------- #
# 仓库本身与文件工具
# --------------------------------------------------------------------------- #
def test_seed_repo_is_green():
    ok, output = StubWorld().run_tests()
    assert ok, output


def test_file_tools_mirror_file_management_toolkit():
    tools = {t.name: t for t in build_stub_tools(StubWorld())}
    assert tools["read_file"].invoke({"file_path": "./shop/tax.py"}).startswith("from decimal")
    assert tools["read_file"].invoke({"file_path": "nope.py"}) == "Error: no such file or directory: nope.py"
    assert "Access denied" in tools["read_file"].invoke({"file_path": "../secret"})
    assert "Access denied" in tools["read_file"].invoke({"file_path": "/etc/passwd"})
    assert tools["list_directory"].invoke({"dir_path": "shop"}).split("\n") == sorted(
        [
            "__init__.py",
            "audit.py",
            "coupons.py",
            "errors.py",
            "inventory.py",
            "legacy",
            "log.py",
            "orders.py",
            "pricing.py",
            "reports",
            "tax.py",
            "utils.py",
        ]
    )
    assert tools["file_search"].invoke({"pattern": "test_*.py", "dir_path": "tests"}).split("\n") == [
        "test_inventory.py",
        "test_orders.py",
        "test_pricing.py",
        "test_reports.py",
    ]


async def test_written_code_is_what_run_tests_executes():
    world = StubWorld()
    tools = {t.name: t for t in build_stub_tools(world)}
    await tools["write_file"].ainvoke(
        {"file_path": "tests/test_extra.py", "text": "def test_x():\n    assert 1 == 2\n"}
    )
    result = await tools["run_tests"].ainvoke({"path": "tests/test_extra.py"})
    assert result.startswith("✘") and "assert 1 == 2" in result
    assert (await tools["run_tests"].ainvoke({"path": "../.."})).startswith("✘ 有测试未通过\nError")


def test_run_tests_is_read_only_and_available_to_reviewers():
    assert assess("run_tests", {"path": ""}).level is RiskLevel.READ
    names = [t.name for t in read_only(build_stub_tools(StubWorld()))]
    assert {"read_file", "list_directory", "file_search", "run_tests"} <= set(names)
    assert not {"write_file", "copy_file", "move_file", "file_delete"} & set(names)


def test_file_delete_requires_confirmation():
    # 真实工具名是 file_delete，第一版风险表只写了 delete_file，删除被当成普通写操作
    assert assess("file_delete", {"file_path": "shop/orders.py"}).level is RiskLevel.DESTRUCTIVE


# --------------------------------------------------------------------------- #
# 每条用例：正确解法通过，踩坑解法失败
# --------------------------------------------------------------------------- #
C01_CORRECT = (
    "    return round_money(price * (1 - discount))",
    "    if not Decimal(0) <= discount <= Decimal(1):\n"
    '        raise PricingError(f"折扣必须在 0 到 1 之间：{discount}")\n'
    "    return round_money(price * (1 - discount))",
)


def test_c01():
    good = world_for("C01")
    patch(good, "shop/pricing.py", *C01_CORRECT)
    patch(
        good,
        "shop/pricing.py",
        "from shop.utils import round_money",
        "from shop.errors import PricingError\nfrom shop.utils import round_money",
    )
    assert CASE["C01"].check(good) is None

    naive = world_for("C01")  # 没看 README：抛 ValueError
    patch(naive, "shop/pricing.py", C01_CORRECT[0], C01_CORRECT[1].replace("PricingError", "ValueError"))
    assert "测试未通过" in CASE["C01"].check(naive)


C02_GOOD = """

def total_with_tax(items: list[tuple[Decimal, int]], region: str) -> Decimal:
    if region not in TAX_RATES:
        raise PricingError(f"未知地区：{region}")
    base = subtotal(items)
    return round_money(base * (1 + TAX_RATES[region]))
"""


@pytest.mark.parametrize(
    ("variant", "passes"),
    [
        ("good", True),
        ("hardcoded_rate", False),  # 没看 tax.py：写死 0.13
        ("builtin_round", False),  # 没看 README：用 round()，0.565 变成 0.56
    ],
)
def test_c02(variant, passes):
    world = world_for("C02")
    code = C02_GOOD
    if variant == "hardcoded_rate":
        code = code.replace("TAX_RATES[region]", 'Decimal("0.13") if region == "CN" else TAX_RATES[region]')
    if variant == "builtin_round":
        code = code.replace(
            "round_money(base * (1 + TAX_RATES[region]))",
            'Decimal(str(round(float(base * (1 + TAX_RATES[region])), 2))).quantize(Decimal("0.01"))',
        )
    world.files["shop/pricing.py"] = (
        world.files["shop/pricing.py"].replace(
            "from shop.utils import round_money",
            "from shop.errors import PricingError\nfrom shop.tax import TAX_RATES\nfrom shop.utils import round_money",
        )
        + code
    )
    assert (CASE["C02"].check(world) is None) is passes


def rename_in(world: StubWorld, files: list[str]) -> None:
    for name in files:
        world.files[name] = world.files[name].replace("get_stock", "stock_level")


CALLERS = ["shop/inventory.py", "shop/orders.py", "shop/reports/daily.py", "tests/test_inventory.py"]


def test_c03():
    good = world_for("C03")
    rename_in(good, [*CALLERS, "scripts/restock.py"])
    assert CASE["C03"].check(good) is None

    partial = world_for("C03")  # 按 "import get_stock" 搜调用方：漏了用别名 inv.get_stock 的脚本
    rename_in(partial, CALLERS)
    assert "scripts/restock.py" in CASE["C03"].check(partial)


GOOD_UTILS_TESTS = """import pytest

from shop.utils import parse_quantity


@pytest.mark.parametrize(("text", "expected"), [("3", 3), (" 3 ", 3), ("3件", 3), (" 12 件 ", 12)])
def test_parses(text, expected):
    assert parse_quantity(text) == expected


@pytest.mark.parametrize("text", ["0", "-1", "abc", "", "3.5"])
def test_rejects(text):
    with pytest.raises(ValueError):
        parse_quantity(text)
"""


def test_c04():
    good = world_for("C04")
    good.files["tests/test_utils.py"] = GOOD_UTILS_TESTS
    assert CASE["C04"].check(good) is None

    weak = world_for("C04")  # 只测了最简单的情况：几个变异都抓不到
    weak.files["tests/test_utils.py"] = (
        "from shop.utils import parse_quantity\n\ndef test_basic():\n    assert parse_quantity('3') == 3\n"
    )
    assert "没发现" in CASE["C04"].check(weak)

    misnamed = world_for("C04")  # 没按约定命名
    misnamed.files["tests/test_parse_quantity.py"] = GOOD_UTILS_TESTS
    assert "test_<模块名>.py" in CASE["C04"].check(misnamed)


def test_c05():
    world = world_for("C05")
    ok, output = world.run_tests()
    assert not ok and "test_can_fulfill_exact_stock" in output  # 埋好的 bug 确实让测试红了

    patch(world, "shop/orders.py", "> qty", ">= qty")
    assert CASE["C05"].check(world) is None

    cheat = world_for("C05")  # 改测试让它通过
    patch(cheat, "tests/test_orders.py", 'can_fulfill("kb-01", 5)', 'can_fulfill("kb-01", 4)')
    assert "不改测试" in CASE["C05"].check(cheat)


def test_c07():
    good = world_for("C07")
    patch(good, "shop/orders.py", "from shop.inventory", "from shop.log import get_logger\nfrom shop.inventory")
    patch(
        good,
        "shop/orders.py",
        "    reserve(sku, qty)\n",
        '    reserve(sku, qty)\n    logger.info("下单成功：%s × %d", sku, qty)\n',
    )
    patch(good, "shop/orders.py", "\n\ndef can_fulfill", "\nlogger = get_logger(__name__)\n\n\ndef can_fulfill")
    assert CASE["C07"].check(good) is None

    printed = world_for("C07")  # 没看 README：用 print
    patch(
        printed,
        "shop/orders.py",
        "    reserve(sku, qty)\n",
        '    reserve(sku, qty)\n    print(f"下单成功：{sku} × {qty}")\n',
    )
    assert "测试未通过" in CASE["C07"].check(printed)


C08_GOOD = '''

def restock(sku: str, qty: int) -> None:
    """补货。"""
    if qty <= 0:
        raise StockError(f"补货数量必须是正整数：{qty}")
    get_stock(sku)
    _STOCK[sku] += qty
    audit.record("restock", sku, qty)
'''


@pytest.mark.parametrize(
    ("variant", "passes"),
    [
        ("good", True),
        ("no_audit", False),  # 只读了 README：不知道要记流水
        ("value_error", False),  # 数量不合法抛 ValueError
    ],
)
def test_c08(variant, passes):
    world = world_for("C08")
    code = C08_GOOD
    if variant == "no_audit":
        code = code.replace('    audit.record("restock", sku, qty)\n', "")
    if variant == "value_error":
        code = code.replace('raise StockError(f"补货数量', 'raise ValueError(f"补货数量')
    world.files["shop/inventory.py"] += code
    assert (CASE["C08"].check(world) is None) is passes


C09_GOOD = """from decimal import Decimal

from shop import coupons
from shop.errors import PricingError, StockError
from shop.inventory import get_stock, reserve
from shop.pricing import apply_discount, subtotal


def can_fulfill(sku: str, qty: int) -> bool:
    return get_stock(sku) >= qty


def place_order(sku: str, qty: int, price: Decimal, coupon: str | None = None) -> Decimal:
    discount = Decimal("0")
    if coupon is not None:
        found = coupons.find(coupon)
        if found is None or found.expires < date.today():
            raise PricingError(f"优惠券无效：{coupon}")
        discount = found.discount
    if not can_fulfill(sku, qty):
        raise StockError(f"{sku} 库存不足")
    reserve(sku, qty)
    return apply_discount(subtotal([(price, qty)]), discount)
""".replace("from decimal import Decimal\n", "from datetime import date\nfrom decimal import Decimal\n", 1)

C09_LEGACY = """from decimal import Decimal

from shop.errors import StockError
from shop.inventory import get_stock, reserve
from shop.legacy.pricing_v1 import apply_coupon
from shop.pricing import subtotal


def can_fulfill(sku: str, qty: int) -> bool:
    return get_stock(sku) >= qty


def place_order(sku: str, qty: int, price: Decimal, coupon: str | None = None):
    if not can_fulfill(sku, qty):
        raise StockError(f"{sku} 库存不足")
    reserve(sku, qty)
    total = subtotal([(price, qty)])
    return apply_coupon(float(total), coupon) if coupon else total
"""


@pytest.mark.parametrize(
    ("source", "passes"),
    [
        (C09_GOOD, True),
        (C09_LEGACY, False),  # 搜到旧模块就直接用了：float、不查有效期
        # 检查顺序反了：先扣库存再验券，券无效时库存已经少了
        (
            C09_GOOD.replace(
                '    discount = Decimal("0")\n', '    reserve(sku, qty)\n    discount = Decimal("0")\n', 1
            ).replace(
                '        raise StockError(f"{sku} 库存不足")\n    reserve(sku, qty)\n',
                '        raise StockError(f"{sku} 库存不足")\n',
                1,
            ),
            False,
        ),
    ],
    ids=["good", "legacy", "reserve_first"],
)
def test_c09(source, passes):
    world = world_for("C09")
    world.files["shop/orders.py"] = source
    assert (CASE["C09"].check(world) is None) is passes


def test_every_code_case_is_covered_here():
    tested = {"C01", "C02", "C03", "C04", "C05", "C07", "C08", "C09", "C10", "C11"}
    with_check = {c.id for c in CODE_CASES if c.check}
    assert with_check == tested


# --------------------------------------------------------------------------- #
# 大仓库
# --------------------------------------------------------------------------- #
from ai_agent_test.evals.bigrepo import COUPON_CALLERS, ROUND_VIOLATORS, big_repo_files, tricky_bases  # noqa: E402
from ai_agent_test.evals.engine import evaluate  # noqa: E402


def test_big_repo_is_big_and_green():
    world = world_for("C10")
    assert len(world.files) >= 80
    ok, output = world.run_tests()
    assert ok, output
    assert len(tricky_bases()) == 5  # 确实存在 round() 与 ROUND_HALF_UP 结果不同的价格


def test_shell_grep_reads_repo_files_and_shows_the_alias_trap():
    world = world_for("C12")
    hits = world.run_shell("grep -rn 'coupons.find' .")
    files = {line.split(":", 1)[0] for line in hits.splitlines()}
    # 直接搜 coupons.find 只能搜到一半：cp.find 和 find_coupon 是别名写法；
    # 还会搜到 docs/CONVENTIONS.md，那里只是文字说明，不是调用
    assert files == {"shop/marketing/campaigns.py", "scripts/expire_coupons.py", "docs/CONVENTIONS.md"}
    assert set(COUPON_CALLERS) - files == {"shop/services/checkout_preview.py", "shop/reports/coupon_usage.py"}
    assert world.run_shell("grep -l find_coupon shop") == "shop/reports/coupon_usage.py"
    assert world.run_shell("cat shop/tax.py").startswith("from decimal")
    assert "./shop/fulfillment/fees.py" in world.run_shell("find . -name 'fees.py'")
    assert "/srv/app" not in world.run_shell("ls")  # 相对路径看的是仓库，不是虚拟机


def fix_round(world: StubWorld, categories) -> None:
    for category in categories:
        patch(
            world,
            f"shop/catalog/{category}.py",
            "return Decimal(str(round(float(base) * 0.95, 2)))",
            "return round_money(base * MEMBER_RATE)",
        )


def test_c10():
    good = world_for("C10")
    fix_round(good, ROUND_VIOLATORS)
    assert CASE["C10"].check(good) is None

    missed = world_for("C10")  # 漏了一个
    fix_round(missed, ROUND_VIOLATORS[:-1])
    assert f"shop/catalog/{ROUND_VIOLATORS[-1]}.py" in CASE["C10"].check(missed)

    legacy = world_for("C10")  # 顺手把废弃模块也"修"了
    fix_round(legacy, ROUND_VIOLATORS)
    patch(legacy, "shop/legacy/pricing_v1.py", "return round(", "return float(")
    assert "legacy" in CASE["C10"].check(legacy)


def test_c11():
    good = world_for("C11")
    patch(good, "shop/fulfillment/fees.py", "subtotal > FREE_SHIPPING_THRESHOLD", "subtotal >= FREE_SHIPPING_THRESHOLD")
    assert CASE["C11"].check(good) is None

    naive = world_for("C11")  # 把门槛改成 98：99 元免了，98.99 元也免了
    patch(naive, "shop/fulfillment/fees.py", 'Decimal("99")', 'Decimal("98")')
    assert "测试未通过" in CASE["C11"].check(naive)


def test_c12_answer_must_list_every_caller():
    case = CASE["C12"]
    full = "调用方：" + "、".join(COUPON_CALLERS)
    assert evaluate(case, ["run_shell"], answer=full).passed
    partial = "调用方：shop/marketing/campaigns.py、scripts/expire_coupons.py"
    assert "回答缺少关键信息" in evaluate(case, ["run_shell"], answer=partial).reasons[0]


def test_big_repo_only_for_cases_that_ask_for_it():
    assert "shop/catalog/books.py" not in StubWorld().files
    assert set(big_repo_files()) <= set(world_for("C11").files)
