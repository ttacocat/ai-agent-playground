from shop.reports.daily import low_stock


def test_low_stock():
    assert low_stock() == ["ms-02"]
    assert low_stock(threshold=6) == ["kb-01", "ms-02"]
