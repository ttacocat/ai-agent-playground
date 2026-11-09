# 开发约定（详细版）

## 库存

- 库存的任何变动（扣减、补货、盘点）都必须调用 `shop.audit.record(action, sku, delta)` 记一笔流水。`delta` 是变动量，扣减为负数。
- 数量参数必须是正整数，否则抛 `StockError`。
- 商品不存在时抛 `StockError`。

## 下单

- 下单失败（库存不足、优惠券无效等）时不能扣减库存：先做完所有检查，再扣库存。

## 优惠券

- 优惠券定义在 `shop/coupons.py`，用 `coupons.find(code)` 查询。
- 不存在或已过期（`expires` 早于今天）的优惠券抛 `PricingError`。
- 优惠按小计计算，用 `pricing.apply_discount`。

## 旧代码

- `shop/legacy/` 下是兼容旧报表的代码，已废弃。新功能不要引用，也不要修改。
- `scripts/` 下的命令行脚本可以用 `print` 输出结果。
