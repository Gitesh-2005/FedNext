import argparse
import logging
import os
import sys
import time
import random
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.federated.client import FederatedClient
from src.network.client_service import FederatedClientService
from src.data.dataset import Vocabulary, load_client_datasets

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | [%(process)d] | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)

def main():
    parser = argparse.ArgumentParser(description="Start Federated Learning GRPC Client")
    parser.add_argument("--id", type=str, required=True, help="Client ID to register as e.g. client_0")
    parser.add_argument("--server", type=str, default="localhost:50051", help="Target gRPC server")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config")
    parser.add_argument("--simulate-failure", action="store_true", help="Simulate dropped packets and slow devices securely")
    args = parser.parse_args()
    
    setup_logging("DEBUG")
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Start gRPC binding
    service = FederatedClientService(client_id=args.id, server_address=args.server)
    
    # 2. Local Data Preparation
    logger.info("Initializing vocabulary and dataset for local partition...")
    vocab = Vocabulary(vocab_size=cfg["model"]["vocab_size"])
    # We must build the full vocab so all clients map identical indices.
    # In a real environment, the server broadcasts `vocab.json`.
    federation_dir = cfg["data"]["federation_dir"]
    txt_files = [f for f in os.listdir(federation_dir) if f.endswith(".txt")]
    all_text = []
    for fname in txt_files:
        with open(os.path.join(federation_dir, fname), encoding="utf-8") as fh:
            all_text.append(fh.read())
    vocab.build(all_text)
    
    client_datasets = load_client_datasets(
        federation_dir=federation_dir,
        vocab=vocab,
        seq_len=cfg["model"]["seq_len"],
        val_fraction=0.0,
        test_fraction=cfg["data"]["test_fraction"],
        min_samples=cfg["data"]["min_samples"],
        non_iid=cfg["federation"].get("non_iid", False),
        num_clients=cfg["federation"].get("num_clients", None),
        dirichlet_alpha=cfg["federation"].get("dirichlet_alpha", 0.5),
        seed=cfg["training"]["seed"],
        cache_dir="data/cache"
    )
    
    if args.id not in client_datasets:
        logger.error(f"Client ID '{args.id}' not found in partitioned datasets! Available: {list(client_datasets.keys())[:5]}...")
        service.close()
        return

    train_ds, val_ds, test_ds = client_datasets[args.id]
    batch_size = cfg["federation"]["local_batch_size"]
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
    
    # Init actual FL Client Logic containing PyTorch optimizer loops
    fl_client = FederatedClient(args.id, train_loader, test_loader, device)
    
    # 3. Register
    samples_count = len(train_ds)
    last_round_seen = 0
    service.register(sample_count=samples_count)
    
    # Phase 8: Initialize TensorBoard Writer for Client local metrics
    writer = SummaryWriter(log_dir=f"logs/client_{args.id}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    m = cfg["model"]
    model_cfg = {
        "vocab_size": m["vocab_size"],
        "embedding_dim": m["embedding_dim"],
        "hidden_dim": m["hidden_dim"],
        "num_layers": m["num_layers"],
        "dropout": m["dropout"],
        "rnn_type": m.get("rnn_type", "LSTM"),
        "bidirectional": m.get("bidirectional", False)
    }
    
    try:
        # Loop for rounds 
        while True:
            logger.info("Awaiting new global round...")
            round_num, global_weights, metadata = service.get_model()
            
            if round_num == -1:
                logger.info("Server signals training is complete (-1). Stopping client.")
                break
                
            if round_num > last_round_seen:
                logger.info(f"Retrieved global model for round {round_num}. Starting local epochs.")
                
                # Client Failure Simulation Logic for Async testing
                if args.simulate_failure:
                    chance = random.random()
                    if chance < 0.2:
                        logger.warning(f"[SIMULATED FAILURE] Client crashed/battery died. Skipping round {round_num}.")
                        time.sleep(random.randint(5, 10))
                        last_round_seen = round_num # Skip this round entirely
                        continue # Network drops entirely
                    elif chance < 0.5:
                        delay = random.randint(3, 8)
                        logger.info(f"[SIMULATED LAG] Client slowing down computations by {delay}s...")
                        time.sleep(delay)

                # Execute PyTorch local SGD Loop
                updated_weights, n_samples, avg_loss = fl_client.local_train(
                    global_state_dict=global_weights,
                    model_cfg=model_cfg,
                    local_epochs=cfg["federation"]["local_epochs"],
                    lr=cfg["training"]["lr"],
                    weight_decay=cfg["training"]["weight_decay"],
                    grad_clip=cfg["training"]["grad_clip"],
                    privacy_cfg=cfg.get("privacy", {"dp_enabled": False})
                )
                
                # Local Evaluation
                test_loss, accuracy = fl_client.evaluate(updated_weights, model_cfg)
                
                # Phase 8: Log to TensorBoard
                writer.add_scalar('Local/Train_Loss', avg_loss, round_num)
                writer.add_scalar('Local/Test_Loss', test_loss, round_num)
                writer.add_scalar('Local/Accuracy', accuracy, round_num)
                writer.add_scalar('System/Samples_Used', n_samples, round_num)
                
                # Push Learned Gradient Weights to Server with Sparsification Filtering
                service.send_update(
                    training_round=round_num,
                    weights=updated_weights,
                    samples_used=n_samples,
                    loss=avg_loss,
                    metrics={'accuracy': accuracy, 'test_loss': test_loss},
                    sparse_threshold=0.005 # Filters out weights under 0.005
                )
                last_round_seen = round_num
                
            time.sleep(3) # Polling gap
            
    except KeyboardInterrupt:
        logger.info("Client shutting down.")
    finally:
        writer.close() # Phase 8 cleanup
        service.close()

if __name__ == "__main__":
    main()
