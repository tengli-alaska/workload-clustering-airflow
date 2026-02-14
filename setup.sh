#!/bin/bash
set -e

echo "=== Server Workload Clustering Pipeline Setup ==="

# Create required Airflow directories
echo "[1/4] Creating directories..."
mkdir -p ./dags ./logs ./plugins ./config ./dags/data ./dags/model ./dags/src

# Set Airflow UID
echo "[2/4] Setting Airflow user..."
echo "AIRFLOW_UID=$(id -u)" > .env

# Verify project structure
echo "[3/4] Verifying project structure..."
for f in dags/airflow.py dags/src/lab.py dags/src/__init__.py dags/data/server_workloads.csv dags/data/test.csv docker-compose.yaml; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done

echo "[4/4] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. docker compose up airflow-init"
echo "  2. docker compose up"
echo "  3. Visit localhost:8080 (airflow2/airflow2)"
echo "  4. Trigger the Server_Workload_Clustering DAG"
echo "  5. docker compose down (when finished)"
