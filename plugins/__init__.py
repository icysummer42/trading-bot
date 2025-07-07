"""Minimal plugin stubs so `signal_generator.py` can import them.
Each plugin now accepts a Config object in its constructor.
Replace method bodies with real logic later.
"""

class UnusualOptionsFlowPlugin:  # noqa: D101
    def __init__(self, cfg):
        self.cfg = cfg

    def fetch(self, *args, **kwargs):
        return {}


class CyberSecurityBreachPlugin:  # noqa: D101
    def __init__(self, cfg):
        self.cfg = cfg

    def fetch(self, *args, **kwargs):
        return {}


class TopTierReleasePlugin:  # noqa: D101
    def __init__(self, cfg):
        self.cfg = cfg

    def fetch(self, *args, **kwargs):
        return {}
