"""
Persistent Accounts Store for LazySD UI.

Manages guest and host accounts stored locally on the host computer
in `models/accounts/accounts.json`.

All accounts, passwords/tokens, and permissions reside exclusively
on the host machine.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_FILE = PROJECT_ROOT / "models" / "accounts" / "accounts.json"


class AccountsManager:
    """Manages persistent guest accounts and token validation."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or ACCOUNTS_FILE

    def _ensure_dir(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> dict:
        self._ensure_dir()
        if not self.storage_path.is_file():
            initial = {
                "version": 1,
                "accounts": [],
            }
            self.save_data(initial)
            return initial
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "accounts": []}

    def save_data(self, data: dict) -> None:
        self._ensure_dir()
        temp_file = self.storage_path.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_file.replace(self.storage_path)

    def list_accounts(self) -> list[dict]:
        data = self.load_data()
        return data.get("accounts", [])

    def get_account(self, account_id: str) -> Optional[dict]:
        for acc in self.list_accounts():
            if acc.get("id") == account_id:
                return acc
        return None

    def find_by_token(self, token: str) -> Optional[dict]:
        if not token:
            return None
        for acc in self.list_accounts():
            if acc.get("is_active", True) and secrets.compare_digest(acc.get("token", ""), token):
                return acc
        return None

    def _hash_password(self, password: str, salt: str) -> str:
        import hashlib
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()

    def create_guest_account(self, username: str, password: Optional[str] = None) -> dict:
        username = (username or "").strip() or "Гость"
        # Sanitize username (letters, numbers, underscores, spaces, hyphens)
        clean_name = "".join(c for c in username if c.isalnum() or c in (" ", "_", "-", "(", ")", ".", "@")).strip()
        if not clean_name:
            clean_name = "Гость"

        data = self.load_data()
        accounts = data.setdefault("accounts", [])

        account_id = f"guest_{secrets.token_hex(4)}"
        token = f"gst_{secrets.token_urlsafe(16)}"

        raw_pwd = (password or "").strip()
        is_custom = bool(raw_pwd)
        if not raw_pwd:
            # Generate friendly initial temporary password
            raw_pwd = f"guest_{secrets.token_hex(3)}"

        salt = secrets.token_hex(16)
        pwd_hash = self._hash_password(raw_pwd, salt)

        new_acc = {
            "id": account_id,
            "username": clean_name,
            "role": "guest",
            "token": token,
            "salt": salt,
            "password_hash": pwd_hash,
            "initial_password": raw_pwd if not is_custom else raw_pwd,
            "has_custom_password": is_custom,
            "created_at": int(time.time()),
            "is_active": True,
        }
        accounts.append(new_acc)
        self.save_data(data)
        return new_acc

    def update_guest_account(
        self,
        account_id: str,
        new_username: Optional[str] = None,
        new_password: Optional[str] = None,
    ) -> Optional[dict]:
        data = self.load_data()
        for acc in data.get("accounts", []):
            if acc.get("id") == account_id:
                if new_username is not None:
                    u = new_username.strip()
                    clean_name = "".join(c for c in u if c.isalnum() or c in (" ", "_", "-", "(", ")", ".", "@")).strip()
                    if clean_name:
                        acc["username"] = clean_name
                if new_password is not None and new_password.strip():
                    raw_pwd = new_password.strip()
                    salt = secrets.token_hex(16)
                    acc["salt"] = salt
                    acc["password_hash"] = self._hash_password(raw_pwd, salt)
                    acc["initial_password"] = raw_pwd
                    acc["has_custom_password"] = True
                    acc["token"] = f"gst_{secrets.token_urlsafe(16)}"
                self.save_data(data)
                return acc
        return None

    def change_guest_password(self, account_id: str, new_password: str) -> bool:
        """Guest self-service: update their own password."""
        raw_pwd = (new_password or "").strip()
        if not raw_pwd:
            return False
        data = self.load_data()
        for acc in data.get("accounts", []):
            if acc.get("id") == account_id:
                salt = secrets.token_hex(16)
                acc["salt"] = salt
                acc["password_hash"] = self._hash_password(raw_pwd, salt)
                acc["initial_password"] = None  # Cleared once guest sets their personal secret
                acc["has_custom_password"] = True
                acc["token"] = f"gst_{secrets.token_urlsafe(16)}"
                self.save_data(data)
                return True
        return False

    def rename_guest_account(self, account_id: str, new_username: str) -> Optional[dict]:
        return self.update_guest_account(account_id, new_username=new_username)

    def delete_guest_account(self, account_id: str) -> bool:
        data = self.load_data()
        accounts = data.get("accounts", [])
        initial_len = len(accounts)
        data["accounts"] = [a for a in accounts if a.get("id") != account_id]
        if len(data["accounts"]) < initial_len:
            self.save_data(data)
            return True
        return False

    def regenerate_guest_token(self, account_id: str) -> Optional[str]:
        data = self.load_data()
        for acc in data.get("accounts", []):
            if acc.get("id") == account_id:
                new_token = f"gst_{secrets.token_urlsafe(16)}"
                acc["token"] = new_token
                self.save_data(data)
                return new_token
        return None

    def verify_login(self, username: str, password: str) -> Optional[dict]:
        """Verify username & password against registered active accounts."""
        if not username or not password:
            return None
        clean_user = username.strip().lower()
        clean_pwd = password.strip()

        for acc in self.list_accounts():
            if not acc.get("is_active", True):
                continue
            acc_user = (acc.get("username") or "").strip().lower()
            if acc_user != clean_user:
                continue

            salt = acc.get("salt")
            pwd_hash = acc.get("password_hash")
            if salt and pwd_hash:
                test_hash = self._hash_password(clean_pwd, salt)
                if secrets.compare_digest(test_hash, pwd_hash):
                    return acc
            elif acc.get("initial_password") and secrets.compare_digest(acc.get("initial_password"), clean_pwd):
                return acc
            elif acc.get("token") and secrets.compare_digest(acc.get("token"), clean_pwd):
                return acc
        return None


# Global singleton instance
_ACCOUNTS_MANAGER = AccountsManager()


def get_accounts_manager() -> AccountsManager:
    return _ACCOUNTS_MANAGER


def resolve_account_from_scope(scope) -> dict:
    """Determine the active caller account & role from ASGI scope."""
    from .share import is_remote_scope, extract_token
    from .settings import get_settings

    # 1. Local loopback defaults to Host (Пользовательский / Владелец)
    if not is_remote_scope(scope):
        return {
            "role": "host",
            "id": "host",
            "username": "Хост",
            "is_guest": False,
        }

    # 2. Remote scope — check token
    provided, _ = extract_token(scope)
    settings = get_settings()

    # Remote admin access: if provided token matches host's share_token
    # AND share_allow_admin is True, caller is recognized as Host
    if provided and settings.share_token and secrets.compare_digest(provided, settings.share_token):
        if settings.share_allow_admin:
            return {
                "role": "host",
                "id": "host_remote",
                "username": "Хост (Удалённый)",
                "is_guest": False,
            }

    # Check registered persistent guest accounts
    if provided:
        guest_acc = _ACCOUNTS_MANAGER.find_by_token(provided)
        if guest_acc:
            return {
                "role": "guest",
                "id": guest_acc["id"],
                "username": guest_acc.get("username", "Гость"),
                "is_guest": True,
            }

    # Fallback if no matching token
    return {
        "role": "anonymous",
        "id": "anonymous",
        "username": "Неавторизован",
        "is_guest": True,
    }
