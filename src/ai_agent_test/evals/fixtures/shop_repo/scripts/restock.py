"""运维脚本：列出需要补货的商品。用法：python scripts/restock.py"""

import shop.inventory as inv
from shop.reports.daily import low_stock


def main() -> None:
    for sku in low_stock():
        print(f"{sku}：剩余 {inv.get_stock(sku)}")


if __name__ == "__main__":
    main()
