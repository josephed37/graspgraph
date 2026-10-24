import pytest

# Filled in by test_robot.py so the success rate shows up in every pytest run's summary.
SUCCESS_RATE_KEY = pytest.StashKey[tuple[int, int]]()


def pytest_terminal_summary(terminalreporter, config):
    if SUCCESS_RATE_KEY in config.stash:
        successes, total = config.stash[SUCCESS_RATE_KEY]
        terminalreporter.write_sep("=", f"scripted pick-and-place: {successes}/{total} succeeded ({successes / total:.0%})")
