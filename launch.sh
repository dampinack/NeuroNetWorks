#!/usr/bin/env bash
# NeuroNetWorks - Application Suite Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$DIR/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "🧠 Launching NeuroNetWorks Deep Learning Suite..."
exec "$PYTHON_EXEC" "$DIR/app.py" "$@"
