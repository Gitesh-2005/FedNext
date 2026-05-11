import io
import json
import logging
import os
import sys
import threading
import time
from concurrent import futures

import grpc
import jwt
import torch
from torch.utils.tensorboard import SummaryWriter

# Append local dir to path to allow the generated pb2 modules to resolve each other
sys.path.append(os.path.dirname(__file__))

import protocol_pb2
import protocol_pb2_grpc
from src.federated.aggregation import federated_average, adaptive_federated_average, fed_async
from src.network.compression import compress_weights, decompress_weights
from src.utils.encryption import AESCipher

logger = logging.getLogger(__name__)

class FederatedLearningServicer(protocol_pb2_grpc.FederatedLearningServicer):
    """
    gRPC Servicer that acts as the central server node.
    It holds the current state of the global model and aggregates updates.
    """
    def __init__(self, initial_model_state: dict, min_clients: int = 2, total_rounds: int = 10, aggregation_method: str = "fedavg", jwt_secret: str = "federated_secret_key"):
        self.lock = threading.Lock()
        self.current_round = 1
        self.total_rounds = total_rounds
        self.min_clients = min_clients
        self.aggregation_method = aggregation_method.lower()
        self.jwt_secret = jwt_secret
        self.cipher = AESCipher(self.jwt_secret)
        
        # Phase 8: Monitoring & Experiment Tracking
        self.writer = SummaryWriter(log_dir=f"logs/server_{time.strftime('%Y%m%d-%H%M%S')}")
        
        self.clients = {}
        self.updates = []
        
        self.global_state_dict = initial_model_state
        self.global_weights_bytes = self.cipher.encrypt(compress_weights(initial_model_state, use_fp16=True, sparse_threshold=0.0))

    def _verify_client(self, context) -> bool:
        """Phase 5: Secure Client Authentication via JWT."""
        metadata = dict(context.invocation_metadata())
        auth_header = metadata.get('authorization', '')
        if not auth_header.startswith('Bearer '):
            return False
            
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return True
        except jwt.InvalidTokenError:
            return False

    def RegisterClient(self, request, context):
        if not self._verify_client(context):
            return protocol_pb2.RegisterResponse(success=False, message="Unauthorized Client Signature")
            
        with self.lock:
            self.clients[request.client_id] = {
                'samples': request.sample_count,
                'last_seen': time.time(),
                'status': 'online'
            }
            logger.info(f"[Server] Client {request.client_id} registered. Total clients: {len(self.clients)}")
        return protocol_pb2.RegisterResponse(
            success=True, 
            message="Registered successfully", 
            current_round=self.current_round
        )

    def GetGlobalModel(self, request, context):
        if not self._verify_client(context):
            return protocol_pb2.ModelResponse(current_round=0, weights=b"", metadata="")
            
        with self.lock:
            if request.client_id in self.clients:
                self.clients[request.client_id]['last_seen'] = time.time()
                
            return protocol_pb2.ModelResponse(
                current_round=self.current_round if self.current_round <= self.total_rounds else -1,
                weights=self.global_weights_bytes,
                metadata=json.dumps({"status": "training" if self.current_round <= self.total_rounds else "finished", "bytes": len(self.global_weights_bytes)})
            )

    def SendLocalUpdate(self, request, context):
        if not self._verify_client(context):
            return protocol_pb2.LocalUpdateResponse(accepted=False, message="Unauthorized Client Signature")
            
        with self.lock:
            # For synchronous modes, strict round checking is enforced.
            # In async mode, we accept stale gradients (up to a limit).
            is_async = (self.aggregation_method == "async")
            
            if not is_async and request.training_round != self.current_round:
                logger.warning(f"[Server] Rejected update from {request.client_id}: round {request.training_round} != server round {self.current_round}")
                return protocol_pb2.LocalUpdateResponse(accepted=False, message="Mismatched round")
            
            if not is_async and any(u['client_id'] == request.client_id for u in self.updates):
                return protocol_pb2.LocalUpdateResponse(accepted=False, message="Duplicate update")

            logger.info(f"[Server] Received update back from {request.client_id} for round {request.training_round} (Size: {len(request.weights)/1024:.2f} KB)")
            
            # Sec Aggregation (Phase 5.1): Decrypt weights from client
            try:
                decrypted_bytes = self.cipher.decrypt(request.weights)
            except Exception as e:
                logger.error(f"[Server] Rejected update from {request.client_id}: Decryption failed. {e}")
                return protocol_pb2.LocalUpdateResponse(accepted=False, message="Decryption failed")
                
            # Decompress and deserialize client weights (Phase 3 Optimization)
            weights = decompress_weights(decrypted_bytes)
            
            metrics = {}
            if request.metrics_json:
                metrics = json.loads(request.metrics_json)
                
            if is_async:
                # Async FL Logic: Apply immediately
                staleness = max(0, self.current_round - request.training_round)
                logger.info(f"[Server] ASYNC update. Staleness={staleness}. Applying FedAsync...")
                
                self.global_state_dict = fed_async(
                    self.global_state_dict,
                    weights,
                    staleness=staleness,
                    base_alpha=0.5,
                    a=2
                )
                self.global_weights_bytes = self.cipher.encrypt(compress_weights(self.global_state_dict, use_fp16=True, sparse_threshold=0.0))
                
                self.updates.append(1) # We use this as a raw counter
                if len(self.updates) >= self.min_clients:
                    # Bump global round equivalent conceptually
                    self.current_round += 1
                    self.updates.clear()
                    
                if self.current_round > self.total_rounds:
                    self._save_final_model()
            else:
                # Sync Aggregation Queue
                self.updates.append({
                    'client_id': request.client_id,
                    'weights': weights,
                    'samples': request.samples_used,
                    'metric': metrics.get('accuracy', 0.0), # For adaptive aggregation
                    'loss': request.loss # Passed from client
                })
                
                # Trigger Aggregation if enough clients responded
                if len(self.updates) >= self.min_clients:
                    self._aggregate_and_step()
                
            return protocol_pb2.LocalUpdateResponse(
                accepted=True,
                message="Update accepted."
            )

    def _save_final_model(self):
        logger.info("=== FEDERATED TRAINING COMPLETE ===")
        self.writer.close() # Phase 8: Clean up TensorBoard handle
        os.makedirs("models", exist_ok=True)
        torch.save({
            'global_state_dict': self.global_state_dict,
            'final_round': self.current_round - 1
        }, "models/global_model_network.pt")
        logger.info("Saved final global model to models/global_model_network.pt")

    def _aggregate_and_step(self):
        logger.info(f"[Server] Aggregating round {self.current_round} with {len(self.updates)} updates...")
        
        # Prepare for aggregation based on method
        agg_input = []
        for u in self.updates:
            if self.aggregation_method == "adaptive":
                agg_input.append((u['weights'], u['samples'], u['metric']))
            else:
                agg_input.append((u['weights'], u['samples']))
                
        if self.aggregation_method == "adaptive":
            self.global_state_dict = adaptive_federated_average(agg_input)
        else:
            self.global_state_dict = federated_average(agg_input)
            
        # Compress and serialize the aggregated model, then Encrypt (Phase 5.1)
        self.global_weights_bytes = self.cipher.encrypt(compress_weights(self.global_state_dict, use_fp16=True, sparse_threshold=0.0))
        
        # Phase 8: Log Global Metrics per round
        avg_loss = sum(u.get('loss', 0.0) for u in self.updates) / len(self.updates)
        avg_acc = sum(u.get('metric', 0.0) for u in self.updates) / len(self.updates)
        self.writer.add_scalar('Global/Avg_Loss', avg_loss, self.current_round)
        self.writer.add_scalar('Global/Avg_Accuracy', avg_acc, self.current_round)
        self.writer.add_scalar('System/Participating_Clients', len(self.updates), self.current_round)
        logger.info(f"[Server] Round {self.current_round} Aggregated Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")
        
        # Step round
        self.current_round += 1
        self.updates.clear()
        
        if self.current_round > self.total_rounds:
            self._save_final_model()
        else:
            logger.info(f"=== Starting Round {self.current_round} ===")


def serve(initial_state: dict, min_clients: int = 2, rounds: int = 10, port: int = 50051) -> tuple[grpc.Server, FederatedLearningServicer]:
    """Starts the gRPC server asynchronously and returns the active server and servicer logic."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20), options=[
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
    ])
    servicer = FederatedLearningServicer(initial_state, min_clients=min_clients, total_rounds=rounds)
    protocol_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC FL Server listening on port {port} (Min clients per round: {min_clients})")
    
    return server, servicer
