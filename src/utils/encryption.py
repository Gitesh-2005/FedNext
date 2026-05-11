import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# A simple wrapper for AES using PyCryptodome
class AESCipher:
    def __init__(self, key: str):
        # Key must be 16, 24, or 32 bytes long for AES
        self.key = key.encode('utf-8').ljust(32, b'\0')[:32]

    def encrypt(self, raw_bytes: bytes) -> bytes:
        cipher = AES.new(self.key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(raw_bytes, AES.block_size))
        return cipher.iv + ct_bytes

    def decrypt(self, enc_bytes: bytes) -> bytes:
        iv = enc_bytes[:16]
        ct = enc_bytes[16:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), AES.block_size)
