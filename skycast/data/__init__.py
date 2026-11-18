"""Data ingestion, multi-source fusion, 3-tier missing-data imputation, and storage."""

from skycast.data.fusion import (
    CANONICAL_FUSED_COLUMNS,
    DOMAIN_CLAMP_BOUNDS,
    RAW_TRANSPARENCY_COLUMNS,
    SourceFusionError,
    fuse_weather_sources,
    weighted_circular_fuse,
    weighted_linear_fuse,
)
from skycast.data.imputation import (
    ALL_IMPUTED_TARGET_COLUMNS,
    CONTINUOUS_IMPUTE_COLUMNS,
    TIER_1_LINEAR,
    TIER_2_DIURNAL,
    TIER_3_CLIMATOLOGY,
    TIER_NONE,
    ImputationAuditReport,
    ImputationResult,
    impute_weather_dataframe,
)
from skycast.data.storage import (
    SkyCastStorage,
    StorageError,
    dataframe_to_hopsworks_frame,
    dataframe_to_supabase_weather_records,
)

__all__ = [
    "ALL_IMPUTED_TARGET_COLUMNS",
    "CANONICAL_FUSED_COLUMNS",
    "CONTINUOUS_IMPUTE_COLUMNS",
    "DOMAIN_CLAMP_BOUNDS",
    "ImputationAuditReport",
    "ImputationResult",
    "RAW_TRANSPARENCY_COLUMNS",
    "SkyCastStorage",
    "SourceFusionError",
    "StorageError",
    "TIER_1_LINEAR",
    "TIER_2_DIURNAL",
    "TIER_3_CLIMATOLOGY",
    "TIER_NONE",
    "dataframe_to_hopsworks_frame",
    "dataframe_to_supabase_weather_records",
    "fuse_weather_sources",
    "impute_weather_dataframe",
    "weighted_circular_fuse",
    "weighted_linear_fuse",
]
