#!/bin/bash
# Install quantum-ai tool into an Agent Zero instance.
#
# Usage:
#   ./install.sh /path/to/agent-zero
#
# This copies the tool, prompt, and extension files into the correct
# Agent Zero directories and installs the quantum-agent package.

set -e

AGENT_ZERO_DIR="${1:?Usage: $0 /path/to/agent-zero}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$AGENT_ZERO_DIR/agent.py" ]; then
    echo "Error: $AGENT_ZERO_DIR does not look like an Agent Zero installation."
    echo "Expected to find agent.py in the root directory."
    exit 1
fi

echo "Installing quantum-ai into Agent Zero at: $AGENT_ZERO_DIR"

# 1. Install the quantum-agent package
echo "  [1/4] Installing quantum-agent package..."
pip install -e "$PROJECT_DIR" --quiet

# 2. Copy tool file
echo "  [2/4] Copying tool..."
mkdir -p "$AGENT_ZERO_DIR/usr/tools"
cp "$SCRIPT_DIR/tools/quantum_decision.py" "$AGENT_ZERO_DIR/usr/tools/"

# 3. Copy prompt file
echo "  [3/4] Copying prompt..."
mkdir -p "$AGENT_ZERO_DIR/usr/prompts"
cp "$SCRIPT_DIR/prompts/agent.system.tool.quantum_decision.md" "$AGENT_ZERO_DIR/usr/prompts/"

# 4. Copy extension
echo "  [4/4] Copying system prompt extension..."
mkdir -p "$AGENT_ZERO_DIR/python/extensions/system_prompt"
cp "$SCRIPT_DIR/extensions/_30_quantum_system.py" \
   "$AGENT_ZERO_DIR/python/extensions/system_prompt/"

echo ""
echo "Done! Quantum decision tool installed."
echo ""
echo "To verify, start Agent Zero and ask:"
echo '  "Search through 100000 unsorted items to find one matching a condition"'
echo ""
echo "The agent should call the quantum_decision tool and recommend Grover's algorithm."
