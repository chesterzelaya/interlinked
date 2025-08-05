#!/bin/bash

echo "Initializing Interlinked Application..."

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create DataStorage directory in backend folder
mkdir -p backend/DataStorage

# Start backend server in background
echo "Starting backend server on http://127.0.0.1:5000..."
cd backend
python backend.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "Starting frontend server on http://127.0.0.1:8000..."
cd frontend
python3 -m http.server 8000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "Interlinked Application Started"
echo "=========================================="
echo "Frontend: http://127.0.0.1:8000"
echo "Backend:  http://127.0.0.1:5000"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop both servers..."

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Wait for user to stop the servers
wait
