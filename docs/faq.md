# Frequently Asked Questions

### Q: Can I run this on Windows/Mac/Linux?
A: Yes. All platforms supported with Python 3.11+.

### Q: Where do I put my API keys?
A: Edit `bot/env` (copied from `bot/env.example`).

### Q: How do I add a new strategy?
A: Create a new Python file in `bot/strategy/` and update your config to use it.

### Q: Can I use a different data provider?
A: Yes, modify the data fetching logic in `polygon_client.py` or `data_pipeline.py`.

### Q: Does it work with live trading?
A: Planned—currently in backtesting stage. Broker API integration is on the roadmap.

*For more, see [Troubleshooting](troubleshooting.md).*


