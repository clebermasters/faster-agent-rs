#!/bin/bash
# Web scraper skill script

URL="$1"

if [ -z "$URL" ]; then
    echo "Usage: $0 <URL>"
    exit 1
fi

curl -s -L "$URL" | head -100
