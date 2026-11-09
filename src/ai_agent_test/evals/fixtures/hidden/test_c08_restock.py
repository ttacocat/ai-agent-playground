import pytest
from shop import audit
from shop.errors import StockError
from shop.inventory import get_stock, restock


def test_restock_adds_stock():
    restock("ms-02", 4)
    assert get_stock("ms-02") == 4


def test_restock_writes_audit_record():
    # docs/CONVENTIONS.md：库存的任何变动都必须记一笔流水（README 只给了链接）
    restock("kb-01", 3)
    assert any(r["sku"] == "kb-01" and r["delta"] == 3 for r in audit.RECORDS), audit.RECORDS


@pytest.mark.parametrize("qty", [0, -2])
def test_quantity_must_be_positive(qty):
    with pytest.raises(StockError):
        restock("kb-01", qty)
    assert get_stock("kb-01") == 5


def test_unknown_sku():
    with pytest.raises(StockError):
        restock("nope", 1)
