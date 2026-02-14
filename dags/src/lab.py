import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

# Feature columns for server workload clustering
FEATURES = [
    "cpu_percent",
    "memory_percent",
    "disk_io_mbps",
    "network_mbps",
    "runtime_seconds",
    "gpu_util_percent",
]


def load_data():
    """
    Loads server workload telemetry from a CSV file.
    Returns base64-encoded pickled DataFrame (JSON-safe for XCom).
    """
    print("Loading server workload data...")
    csv_path = os.path.join(os.path.dirname(__file__), "../data/server_workloads.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} workload records with columns: {list(df.columns)}")
    serialized = pickle.dumps(df)
    return base64.b64encode(serialized).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data, selects workload features, scales with MinMax,
    and returns base64-encoded pickled numpy array.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    print(f"Preprocessing {len(df)} records...")
    df = df.dropna()

    # Select server workload features for clustering
    clustering_data = df[FEATURES]
    print(f"Feature stats before scaling:\n{clustering_data.describe()}")

    scaler = MinMaxScaler()
    clustering_scaled = scaler.fit_transform(clustering_data)

    serialized = pickle.dumps(clustering_scaled)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Fits KMeans for k=1..20, saves the last model, returns SSE list.
    """
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    max_k = 20  # 20 is enough for workload profiling
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        print(f"k={k:2d}  SSE={kmeans.inertia_:.4f}")

    # Save the last-fitted model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"Model saved to {output_path}")

    return sse  # JSON-serializable list


def load_model_elbow(filename: str, sse: list):
    """
    Loads saved model, finds optimal k via elbow method,
    and classifies test workloads.
    """
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(model_path, "rb"))

    # Find elbow point
    max_k = len(sse) + 1
    kl = KneeLocator(range(1, max_k), sse, curve="convex", direction="decreasing")
    print(f"=== Optimal number of workload clusters: {kl.elbow} ===")

    # Classify test workloads
    test_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")
    df_test = pd.read_csv(test_path)
    print(f"\nTest workloads:\n{df_test}")

    # Scale test data with same feature range (0-1 approx)
    scaler = MinMaxScaler()
    # Refit on test data for prediction (in production you'd save/load the scaler)
    predictions = loaded_model.predict(scaler.fit_transform(df_test[FEATURES]))

    labels = {0: "Profile-A", 1: "Profile-B", 2: "Profile-C", 3: "Profile-D"}
    for i, pred in enumerate(predictions):
        label = labels.get(pred, f"Cluster-{pred}")
        print(f"  Workload {i+1}: cluster={pred} ({label})")

    return int(predictions[0])