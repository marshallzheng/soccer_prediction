from corner_predictor.engine.match_runner import registry
from corner_predictor.persistence.repository import MatchRepository

_repository = MatchRepository()


def get_repository() -> MatchRepository:
    return _repository


def get_registry():
    return registry
