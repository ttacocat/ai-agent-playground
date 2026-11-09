import logging
from decimal import Decimal

from shop.orders import place_order


def test_place_order_logs_sku_and_quantity(caplog, capsys):
    caplog.set_level(logging.INFO)
    place_order("hd-03", 3, Decimal("1"))
    records = [r for r in caplog.records if r.levelno >= logging.INFO and r.name.startswith("shop")]
    assert records, "没有通过 shop 的 logger 记录 INFO 级别日志"
    text = " ".join(r.getMessage() for r in records)
    assert "hd-03" in text and "3" in text
    assert capsys.readouterr().out == "", "README 约定不要用 print"
