# Bedrock Models Tracker

Monitors AWS Bedrock model availability and alerts when new models are released.

## Quick Start

```bash
# Check for new models
python3 skills/bedrock-models/scripts/track_models.py

# View current models
cat skills/bedrock-models/data/bedrock_models.csv
```

## What It Does

- Scrapes AWS Bedrock documentation for available models
- Tracks model history with first-seen timestamps
- Detects and alerts on new model releases
- Exports data to JSON and CSV formats

## Output Files

- `data/bedrock_models_memory.csv` - Historical tracking with timestamps
- `data/bedrock_models.csv` - Current snapshot
- `data/bedrock_models.json` - Current snapshot (JSON)

## Memory System

The skill maintains a persistent memory file that tracks:
- All models ever seen
- First seen timestamp for each model
- Provider, model name, model ID, and capabilities

On each run, it compares current models against memory and reports new additions.