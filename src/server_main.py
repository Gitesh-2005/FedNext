import argparse
import ast
import logging
import sys
import time
import yaml

import torch
from src.models.lstm import LSTMLanguageModel

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)

def main():
    parser = argparse.ArgumentParser(description="Start Federated Learning GRPC Server")
    parser.add_argument("--port", type=int, default=50051, help="gRPC listener port")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config")
    args = parser.parse_args()
    
    setup_logging("INFO")
    logger.info("Starting Federated Server (gRPC)")
    
    cfg = load_config(args.config)
    
    # Initialize the global model so we have weights to serve
    device = torch.device("cpu") # Server keeps model in memory
    m = cfg["model"]
    # Provide actual config to model initialization
    model = LSTMLanguageModel(
        vocab_size=m["vocab_size"],
        embedding_dim=m["embedding_dim"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        rnn_type=m.get("rnn_type", "LSTM"),
        bidirectional=m.get("bidirectional", False)
    ).to(device)
    
    initial_weights = model.state_dict()
    
    from src.network.server_service import serve
    
    min_clients = cfg["federation"].get("clients_per_round", 2)
    total_rounds = cfg["federation"].get("rounds", 10)
    
    server, servicer = serve(initial_weights, min_clients=min_clients, rounds=total_rounds, port=args.port)
    
    logger.info("Server is blocking and listening. Press CTRL+C to quit.")
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Server shut down.")

if __name__ == "__main__":
    main()
