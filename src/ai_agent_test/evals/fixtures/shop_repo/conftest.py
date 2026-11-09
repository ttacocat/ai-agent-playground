import pytest
import shop.audit as audit
import shop.inventory as inventory


@pytest.fixture(autouse=True)
def restore_stock():
    """库存和流水是模块级状态：每个测试结束后还原，测试之间互不影响。"""
    snapshot = dict(inventory._STOCK)
    records = list(audit.RECORDS)
    yield
    inventory._STOCK.clear()
    inventory._STOCK.update(snapshot)
    audit.RECORDS[:] = records
