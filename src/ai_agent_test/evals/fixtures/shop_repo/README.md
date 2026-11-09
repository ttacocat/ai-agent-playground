# shop

小型订单与计价模块。

## 约定

- 金额一律用 `decimal.Decimal`。四舍五入统一调用 `shop.utils.round_money`（ROUND_HALF_UP），不要用内置的 `round()`。
- 业务错误统一抛 `shop.errors` 里的异常：计价问题抛 `PricingError`，库存问题抛 `StockError`。不要直接抛 `ValueError`。
  例外：`shop.utils` 里的输入解析函数（`parse_*`）在格式不对时抛 `ValueError`。
- 日志用 `shop.log.get_logger(__name__)`，不要用 `print`。
- 税率只在 `shop/tax.py` 的 `TAX_RATES` 里维护，代码里不要写死税率。
- 测试放在 `tests/` 下，文件名为 `test_<模块名>.py`。

库存、优惠券、旧代码相关的详细约定见 [docs/CONVENTIONS.md](docs/CONVENTIONS.md)。

## 运行测试

```bash
pytest
```
