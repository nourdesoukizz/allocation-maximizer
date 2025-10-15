#!/bin/bash
set -e

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Frontend build complete!"
echo "Starting backend server..."
python main.py