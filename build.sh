#!/usr/bin/env bash
# ShopSense Production Build Script for Render / Cloud Deployment
# Generates the indexed SQLite catalog database offline during build
# so that the API container boots with zero in-memory metadata overhead.
set -e

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🗄️ Building disk-backed SQLite catalog from metadata parquet..."
python -m src.catalog_db

echo "✅ Build complete! Catalog database ready for lightweight serving."
