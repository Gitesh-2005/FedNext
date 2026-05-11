import io
import json
import logging
import os
import sys
import time

import grpc
import torch
import jwt

# Append local dir to path to allow the generated pb2 modules to resolve each other
sys.path.append(os.path.dirname(__file__))

import protocol_pb2
import protocol_pb2_grpc
from src.network.compression import compress_weights, decompress_weights
from src.utils.encryption import AESCipher

logger = logging.getLogger(__name__)

class FederatedClientService:
    """
    gRPC wrapper for the client node.
    This talks to the remote server to fetch weights and push gradients.
    """
    def __init__(self, client_id: str, server_address: str = "localhost:50051", jwt_secret: str = "federated_secret_key"):
        self.client_id = client_id
        self.server_address = server_address
        
        # Phase 5.1: AES Encryption for Secure Payload
        self.cipher = AESCipher(jwt_secret) # Reuse JWT secret as symmetric key

        
        # Phase 5: Client Authentication via JWT
        self.auth_token = jwt.encode({"client_id": self.client_id}, jwt_secret, algorithm="HS256")
        
        # Allow huge message sizes for ML tensor transmissions
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
        self.channel = grpc.insecure_channel(self.server_address, options=options)
        self.stub = protocol_pb2_grpc.FederatedLearningStub(self.channel)

    def _get_metadata(self):
        return (("authorization", f"Bearer {self.auth_token}"),)

    def register(self, sample_count: int = 0) -> int:
        """Register the client presence on the server."""
        request = protocol_pb2.RegisterRequest(
            client_id=self.client_id,
            sample_count=sample_count
        )
        response = self.stub.RegisterClient(request, metadata=self._get_metadata())
        if response.success:
            logger.info(f"[{self.client_id}] Registered on Server. Round: {response.current_round}")
        else:
            logger.error(f"[{self.client_id}] Registration failed: {response.message}")
        return response.current_round

    def get_model(self) -> tuple[int, dict, dict]:
        """Retrieve global model weights and metadata from server."""
        request = protocol_pb2.ModelRequest(client_id=self.client_id)
        
        start_time = time.time()
        response = self.stub.GetGlobalModel(request, metadata=self._get_metadata())
        latency = time.time() - start_time
        
        if response.current_round != -1:
            dl_kb = len(response.weights) / 1024
            logger.debug(f"[{self.client_id}] Downloaded model: {dl_kb:.2f} KB | Latency: {latency*1000:.1f}ms")
        
        # Sec Aggregation (Phase 5.1): Decrypt weights
        try:
            decrypted_bytes = self.cipher.decrypt(response.weights) if response.weights else b""
        except Exception as e:
            logger.error(f"[{self.client_id}] Failed to decrypt server model. Discarding: {e}")
            decrypted_bytes = b""
            
        # Decompress bytes back into a PyTorch state_dict
        weights = decompress_weights(decrypted_bytes) if decrypted_bytes else {}
        metadata = json.loads(response.metadata) if response.metadata else {}
        
        return response.current_round, weights, metadata

    def send_update(self, training_round: int, weights: dict, samples_used: int, loss: float, metrics: dict, sparse_threshold: float = 0.001) -> bool:
        """Push local training findings up to the server node."""
        
        start_comp = time.time()
        # Sparsify, Quantize down to float16, and Zip
        compressed_bytes = compress_weights(weights, use_fp16=True, sparse_threshold=sparse_threshold)
        
        # Sec Aggregation (Phase 5.1): Encrypt payload before sending
        encrypted_bytes = self.cipher.encrypt(compressed_bytes)
        
        comp_time = time.time() - start_comp
        
        ul_kb = len(encrypted_bytes) / 1024
        logger.info(f"[{self.client_id}] Pushing updates for round {training_round} | Size: {ul_kb:.2f} KB (Enc) | Comp/Enc_time: {comp_time*1000:.1f}ms")
        
        # Track bytes inside the metrics payload natively
        metrics['tx_bytes'] = len(encrypted_bytes)
        metrics['ul_speed_kb_s'] = ul_kb / max(comp_time, 0.001)
        
        request = protocol_pb2.LocalUpdateRequest(
            client_id=self.client_id,
            training_round=training_round,
            samples_used=samples_used,
            loss=loss,
            metrics_json=json.dumps(metrics),
            weights=encrypted_bytes
        )
        
        response = self.stub.SendLocalUpdate(request, metadata=self._get_metadata())
        return response.accepted
        
    def close(self):
        """Disconnect safely."""
        self.channel.close()
