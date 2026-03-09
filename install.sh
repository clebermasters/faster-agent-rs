#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Building Skill Agent...${NC}"

# Build the project in release mode
echo -e "${YELLOW}Compiling in release mode...${NC}"
cargo build --release

# Determine install location
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BINARY_NAME="skill-agent"
SOURCE_BINARY="./target/release/$BINARY_NAME"

if [ ! -f "$SOURCE_BINARY" ]; then
    echo -e "${RED}Error: Build failed - binary not found at $SOURCE_BINARY${NC}"
    exit 1
fi

# Check if we need sudo for installation
if [ -w "$INSTALL_DIR" ]; then
    cp "$SOURCE_BINARY" "$INSTALL_DIR/$BINARY_NAME"
    echo -e "${GREEN}✅ Installed to $INSTALL_DIR/$BINARY_NAME${NC}"
else
    echo -e "${YELLOW}⚠️  Need sudo to install to $INSTALL_DIR${NC}"
    sudo cp "$SOURCE_BINARY" "$INSTALL_DIR/$BINARY_NAME"
    echo -e "${GREEN}✅ Installed to $INSTALL_DIR/$BINARY_NAME (with sudo)${NC}"
fi

echo -e "${GREEN}🎉 Installation complete!${NC}"
echo ""
echo "Usage:"
echo "  skill-agent --help"
echo "  skill-agent agent \"your task here\""
echo ""
echo "Configuration:"
echo "  Create a .env file in your project directory with:"
echo "    LLM_PROVIDER=minimax"
echo "    DEFAULT_MODEL=MiniMax-Text-01"
echo "    MINIMAX_API_KEY=your-api-key"
echo ""
