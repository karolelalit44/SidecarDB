__all__ = ["split_csv"]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
