# Example Skills Directory

This directory contains skills that the agent can discover and use.

## Structure

Each skill is a directory containing:

- `SKILL.md` - Skill definition with frontmatter and instructions
- `scripts/` - Executable scripts (optional)
- `references/` - Documentation files (optional)

## Example Skill Format

```yaml
---
name: My Skill
description: What this skill does
trigger:
  - keyword1
  - keyword2
capabilities:
  - capability 1
  - capability 2
---

# Instructions

Describe how to use this skill...
```
