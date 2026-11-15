"""A0-matched settings; environment configuration is imported, never rewritten."""
from experiments.no_shared_kpi.protocol import TRAIN_SETTINGS, feature_spec
VARIANT = 'no_shared_kpi'
TOTAL_STEPS = 200000
SPECIFICATION = feature_spec(VARIANT)
# Production launch requires the run-specific saved A0 training manifest.
# DOE is OFF, matching the current A0 protocol (not earlier experiments).
