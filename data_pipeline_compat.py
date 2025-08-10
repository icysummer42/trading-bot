"""
Compatibility wrapper for the unified data pipeline
Ensures seamless transition from old DataPipeline to UnifiedDataPipeline
"""

from data_pipeline_unified import UnifiedDataPipeline

# Backward compatibility alias
DataPipeline = UnifiedDataPipeline

# Expose all public classes and functions
from data_pipeline_unified import (
    DataQualityMetrics,
    DataPipelineError,
    DataSourceManager,
    CacheManager,
    DataValidator
)
