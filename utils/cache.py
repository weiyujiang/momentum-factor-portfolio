"""
Caching utilities for efficient data storage and retrieval.
"""

import os
import pickle
import json
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path


def ensure_cache_dir(cache_path: str) -> None:
    """Create cache directory if it doesn't exist."""
    Path(cache_path).mkdir(parents=True, exist_ok=True)


def get_cache_key(params: Dict) -> str:
    """Generate cache key from parameters."""
    # Sort dictionary for consistent key generation
    sorted_params = sorted(params.items())
    return "_".join([f"{k}={v}" for k, v in sorted_params])


def save_to_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file."""
    ensure_cache_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filepath: str) -> Optional[Any]:
    """Load data from pickle file. Returns None if file doesn't exist."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        return None


def save_to_json(data: Dict, filepath: str) -> None:
    """Save dictionary to JSON file."""
    ensure_cache_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_from_json(filepath: str) -> Optional[Dict]:
    """Load dictionary from JSON file. Returns None if file doesn't exist."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {filepath}: {e}")
        return None


def is_cache_valid(filepath: str, max_age_days: int = 7) -> bool:
    """Check if cached file is still valid based on age."""
    if not os.path.exists(filepath):
        return False

    # Get file modification time
    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    age = datetime.now() - file_time

    return age < timedelta(days=max_age_days)


def get_file_age_days(filepath: str) -> float:
    """Get age of file in days."""
    if not os.path.exists(filepath):
        return float('inf')

    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    age = datetime.now() - file_time
    return age.total_seconds() / (24 * 3600)


def clear_cache(cache_dir: str, pattern: str = '*') -> int:
    """
    Clear cache files matching pattern.
    Returns number of files deleted.
    """
    import glob

    files = glob.glob(os.path.join(cache_dir, pattern))
    count = 0
    for file in files:
        try:
            os.remove(file)
            count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    return count


def get_cache_size_mb(cache_dir: str) -> float:
    """Get total size of cache directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    return total_size / (1024 * 1024)  # Convert to MB


class CacheManager:
    """Manage cached data with automatic validation and cleanup."""

    def __init__(self, cache_dir: str, max_age_days: int = 7):
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        ensure_cache_dir(cache_dir)

    def get(self, key: str) -> Optional[Any]:
        """Get cached data if valid, otherwise return None."""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")

        if not is_cache_valid(filepath, self.max_age_days):
            return None

        return load_from_pickle(filepath)

    def set(self, key: str, data: Any) -> None:
        """Save data to cache."""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        save_to_pickle(data, filepath)

    def exists(self, key: str) -> bool:
        """Check if key exists in cache and is valid."""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        return is_cache_valid(filepath, self.max_age_days)

    def delete(self, key: str) -> bool:
        """Delete cached item. Returns True if deleted, False if not found."""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def clear_all(self) -> int:
        """Clear all cached items. Returns number of files deleted."""
        return clear_cache(self.cache_dir, '*.pkl')

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        return {
            'total_files': len(files),
            'size_mb': get_cache_size_mb(self.cache_dir),
            'cache_dir': self.cache_dir,
            'max_age_days': self.max_age_days,
        }
