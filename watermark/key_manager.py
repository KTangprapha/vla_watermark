"""Step 1 — Key Generation.

K = KDF(user_id, salt)   # deterministic, 32-byte master key
K → K_text_seed          # first 16 bytes → text trigger seed
K → K_visual_seed        # bytes 16-32    → visual trigger seed
K → K_sig_seed           # HMAC-derived   → signature seed
K → K_stainlock_seed     # HMAC-derived   → StainLock u/v vectors
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass, field


_ITERATIONS = 100_000
_HASH_ALG   = "sha256"
_KEY_LEN    = 32  # bytes


def _hmac_derive(master_key: bytes, label: str) -> bytes:
    """One-step HMAC-SHA256 key derivation: HMAC(master_key, label)."""
    return hmac.new(master_key, label.encode(), digestmod=hashlib.sha256).digest()


@dataclass(frozen=True)
class KeyBundle:
    """All sub-keys derived from one master key K.

    Attributes
    ----------
    master_key    : 32-byte master secret (never share this)
    user_id       : opaque string identifying the key owner
    salt          : 16-byte random salt used during KDF
    K_text_seed   : int seed for text-trigger RNG  (from first 8 bytes of K)
    K_visual_seed : int seed for visual-trigger RNG (from bytes 8-16 of K)
    K_sig_seed    : int seed for signature RNG (HMAC-derived)
    K_stainlock_seed : int seed for StainLock u/v (HMAC-derived)
    """
    master_key:        bytes
    user_id:           str
    salt:              bytes
    K_text_seed:       int
    K_visual_seed:     int
    K_sig_seed:        int
    K_stainlock_seed:  int

    # human-readable hex of the master key (for logging — treat as secret)
    @property
    def key_hex(self) -> str:
        return self.master_key.hex()

    def summary(self) -> str:
        return (
            f"KeyBundle(user_id={self.user_id!r}, "
            f"salt={self.salt.hex()[:8]}…, "
            f"K_text_seed={self.K_text_seed}, "
            f"K_visual_seed={self.K_visual_seed}, "
            f"K_sig_seed={self.K_sig_seed}, "
            f"K_stainlock_seed={self.K_stainlock_seed})"
        )


class KeyManager:
    """Deterministic key generation for VLA watermarking.

    Usage
    -----
    bundle = KeyManager.generate("user_001")          # random salt
    bundle = KeyManager.generate("user_001", salt)    # reproducible

    Only the holder of `bundle.master_key` + `salt` can regenerate all sub-keys,
    which is sufficient to reconstruct the exact trigger and signature used during
    watermark embedding — enabling ownership proof.
    """

    @staticmethod
    def generate(user_id: str, salt: bytes | None = None) -> KeyBundle:
        """Derive a KeyBundle from user_id and optional salt.

        Parameters
        ----------
        user_id : arbitrary string identifying the model owner
        salt    : 16-byte bytes (generated randomly if None)
        """
        if salt is None:
            salt = os.urandom(16)
        if len(salt) < 8:
            raise ValueError("salt must be at least 8 bytes")

        # Master key: PBKDF2-HMAC-SHA256(password=user_id, salt=salt)
        master_key = hashlib.pbkdf2_hmac(
            _HASH_ALG,
            user_id.encode("utf-8"),
            salt,
            _ITERATIONS,
            dklen=_KEY_LEN,
        )

        # Sub-keys: K[0:8] and K[8:16] directly as little-endian ints
        K_text_seed   = struct.unpack_from("<Q", master_key,  0)[0]
        K_visual_seed = struct.unpack_from("<Q", master_key,  8)[0]

        # Signature and StainLock seeds: HMAC-derived to avoid correlation
        sig_bytes        = _hmac_derive(master_key, "signature")
        stainlock_bytes  = _hmac_derive(master_key, "stainlock")
        K_sig_seed       = struct.unpack_from("<Q", sig_bytes,       0)[0]
        K_stainlock_seed = struct.unpack_from("<Q", stainlock_bytes, 0)[0]

        return KeyBundle(
            master_key       = master_key,
            user_id          = user_id,
            salt             = salt,
            K_text_seed      = K_text_seed,
            K_visual_seed    = K_visual_seed,
            K_sig_seed       = K_sig_seed,
            K_stainlock_seed = K_stainlock_seed,
        )

    @staticmethod
    def from_master_key(master_key: bytes, user_id: str, salt: bytes) -> KeyBundle:
        """Reconstruct a KeyBundle from a saved master key + salt (for detection)."""
        if len(master_key) != _KEY_LEN:
            raise ValueError(f"master_key must be {_KEY_LEN} bytes")

        K_text_seed   = struct.unpack_from("<Q", master_key,  0)[0]
        K_visual_seed = struct.unpack_from("<Q", master_key,  8)[0]

        sig_bytes        = _hmac_derive(master_key, "signature")
        stainlock_bytes  = _hmac_derive(master_key, "stainlock")
        K_sig_seed       = struct.unpack_from("<Q", sig_bytes,       0)[0]
        K_stainlock_seed = struct.unpack_from("<Q", stainlock_bytes, 0)[0]

        return KeyBundle(
            master_key       = master_key,
            user_id          = user_id,
            salt             = salt,
            K_text_seed      = K_text_seed,
            K_visual_seed    = K_visual_seed,
            K_sig_seed       = K_sig_seed,
            K_stainlock_seed = K_stainlock_seed,
        )

    @staticmethod
    def save(bundle: KeyBundle, path: str) -> None:
        """Save master key + salt to a binary file (keep this file secret)."""
        import json, base64
        data = {
            "user_id":    bundle.user_id,
            "salt":       base64.b64encode(bundle.salt).decode(),
            "master_key": base64.b64encode(bundle.master_key).decode(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> KeyBundle:
        """Load a previously saved KeyBundle."""
        import json, base64
        with open(path) as f:
            data = json.load(f)
        master_key = base64.b64decode(data["master_key"])
        salt       = base64.b64decode(data["salt"])
        return KeyManager.from_master_key(master_key, data["user_id"], salt)
