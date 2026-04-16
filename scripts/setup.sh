#!/bin/bash
set -e

echo "=== EMPIRE Setup ==="
echo ""

# Check prerequisites
command -v python3 >/dev/null || { echo "ERROR: python3 required"; exit 1; }
command -v node >/dev/null || { echo "ERROR: node required"; exit 1; }
command -v npm >/dev/null || { echo "ERROR: npm required"; exit 1; }

# Backend
echo "--- Installing backend dependencies ---"
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Dashboard
echo "--- Installing dashboard dependencies ---"
cd dashboard
npm install
cd ..

# Mobile (optional)
echo "--- Installing mobile dependencies ---"
cd mobile
npm install
cd ..

# Environment
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo ">>> Created .env from .env.example"
    echo ">>> Edit .env with your API keys before running!"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run:"
echo "  1. Edit .env with your API keys"
echo "  2. Backend:   cd backend && source venv/bin/activate && python main.py"
echo "  3. Dashboard: cd dashboard && npm run dev"
echo "  4. Mobile:    cd mobile && npx expo start"
echo ""
echo "  Backend runs on http://localhost:8000"
echo "  Dashboard runs on http://localhost:5173"
