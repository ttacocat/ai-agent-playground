import pytest
from shop.errors import StockError
from shop.inventory import get_stock, reserve


def test_get_stock():
    assert get_stock("hd-03") == 12


def test_unknown_sku():
    with pytest.raises(StockError):
        get_stock("nope")


def test_reserve():
    reserve("hd-03", 2)
    assert get_stock("hd-03") == 10
