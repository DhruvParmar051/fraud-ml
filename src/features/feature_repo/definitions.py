"""Feast feature definitions: entity (``nameDest``) and the destination-stats view.

This module is consumed by ``feast apply`` to register entities and feature views
against the offline Parquet source and the Redis online store.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

destination = Entity(
    name="destination",
    join_keys=["nameDest"],
    value_type=ValueType.STRING,
)

destination_stats_source = FileSource(
    name="destination_stats_source",
    path="data/destination_stats.parquet",
    timestamp_field="event_timestamp",
)

destination_stats_fv = FeatureView(
    name="destination_stats",
    entities=[destination],
    ttl=timedelta(days=7),
    schema=[
        Field(name="dest_tx_count_1h", dtype=Int64),
        Field(name="dest_tx_count_24h", dtype=Int64),
        Field(name="dest_tx_count_7d", dtype=Int64),
        Field(name="dest_amount_sum_24h", dtype=Float64),
        Field(name="dest_amount_avg_24h", dtype=Float64),
    ],
    source=destination_stats_source,
    online=True,
)
