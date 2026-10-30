"""Feast entity definitions for real-time fraud feature views."""

from __future__ import annotations

from feast import Entity
from feast.value_type import ValueType

user_entity = Entity(
    name="user_id",
    join_keys=["user_id"],
    value_type=ValueType.STRING,
    description="Unique customer account identifier for rolling spend and velocity features.",
)

merchant_entity = Entity(
    name="merchant_category",
    join_keys=["merchant_category"],
    value_type=ValueType.STRING,
    description="Merchant category classification for historical chargeback risk lookup.",
)
