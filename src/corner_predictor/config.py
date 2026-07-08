from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CORNER_")

    db_path: Path = BASE_DIR / "data" / "corner_predictor.db"

    # Tick loop
    tick_interval_seconds: float = 1.0
    minutes_per_tick: float = 1.0

    # Probability model
    prior_corners_per_90min: float = 10.0
    prior_weight_minutes: float = 20.0
    """Pseudo-minutes of prior weight (k) in the rate shrinkage formula."""
    rate_multiplier_min: float = 0.3
    rate_multiplier_max: float = 3.0
    default_threshold: float = 9.5
    pmf_max_remaining_corners: int = 20
    overdispersion_k: float = 8.0
    """Negative-binomial dispersion placeholder pending real-data calibration."""
    probability_model: str = "poisson"

    # Feature engine
    rolling_window_minutes: float = 10.0
    baseline_attacks_per_min_per_team: float = 50.0 / 90.0
    """Expected 'dangerous attacks' per minute per team at average intensity, used to gauge current pace."""
    intensity_factor_min: float = 0.5
    intensity_factor_max: float = 2.0
    urgency_score_diff_weight: float = 0.15
    """Multiplier added per goal of score differential (trailing/leading teams tend to change attacking approach)."""
    late_game_threshold_minutes: float = 15.0
    late_game_multiplier: float = 1.1


settings = Settings()
