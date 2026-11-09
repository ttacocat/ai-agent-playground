RECORDS: list[dict] = []


def record(action: str, sku: str, delta: int) -> None:
    """记一笔库存流水。delta 是变动量，扣减为负数。"""
    RECORDS.append({"action": action, "sku": sku, "delta": delta})
