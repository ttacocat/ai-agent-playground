from decimal import Decimal

import shop.inventory as inventory
from shop.orders import can_fulfill, place_order


def test_new_name_exists_and_old_name_is_gone():
    assert inventory.stock_level("hd-03") == 12
    assert not hasattr(inventory, "get_stock")


def test_callers_still_work():
    assert can_fulfill("kb-01", 5)
    assert place_order("hd-03", 1, Decimal("10")) == Decimal("10.00")


def test_report_and_script_callers_still_work(capsys):
    import runpy

    from shop.reports.daily import low_stock

    assert low_stock() == ["ms-02"]
    # scripts/restock.py 用的是别名写法 inv.get_stock(...)，按函数名搜 "from ... import get_stock" 搜不到
    runpy.run_path("scripts/restock.py", run_name="__main__")
    assert "ms-02" in capsys.readouterr().out
