"""
Profile manager for loading, saving, and managing radar profiles.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from .radar_profile import RadarProfile


class ProfileManager:
    """
    Manages radar profiles - loading from defaults, saving custom profiles.
    """

    def __init__(self, user_profiles_dir: Optional[str] = None):
        """
        Initialize the profile manager.

        Args:
            user_profiles_dir: Directory for user-created profiles.
                              Defaults to 'user_profiles' in the profiles directory.
        """
        self._profiles: Dict[str, RadarProfile] = {}

        # Get the defaults directory (relative to this file)
        self._defaults_dir = Path(__file__).parent / "defaults"

        # Set up user profiles directory
        if user_profiles_dir:
            self._user_dir = Path(user_profiles_dir)
        else:
            self._user_dir = Path(__file__).parent / "user_profiles"

        # Create user profiles directory if it doesn't exist
        self._user_dir.mkdir(parents=True, exist_ok=True)

        # Load all profiles
        self._load_default_profiles()
        self._load_user_profiles()

    def _load_default_profiles(self):
        """Load all default profiles from the defaults directory."""
        if not self._defaults_dir.exists():
            return

        for yaml_file in self._defaults_dir.glob("*.yaml"):
            try:
                profile = RadarProfile.from_yaml_file(str(yaml_file))
                self._profiles[profile.name] = profile
            except Exception as e:
                print(f"Warning: Could not load default profile {yaml_file}: {e}")

    def _load_user_profiles(self):
        """Load all user profiles from the user profiles directory."""
        if not self._user_dir.exists():
            return

        for yaml_file in self._user_dir.glob("*.yaml"):
            try:
                profile = RadarProfile.from_yaml_file(str(yaml_file))
                # Add (Custom) suffix if name conflicts with default
                if profile.name in self._profiles:
                    profile.name = f"{profile.name} (Custom)"
                self._profiles[profile.name] = profile
            except Exception as e:
                print(f"Warning: Could not load user profile {yaml_file}: {e}")

    def reload_profiles(self):
        """Reload all profiles from disk."""
        self._profiles.clear()
        self._load_default_profiles()
        self._load_user_profiles()

    def get_profile(self, name: str) -> Optional[RadarProfile]:
        """Get a profile by name."""
        return self._profiles.get(name)

    def get_all_profiles(self) -> Dict[str, RadarProfile]:
        """Get all loaded profiles."""
        return self._profiles.copy()

    def get_profile_names(self) -> List[str]:
        """Get list of all profile names."""
        return list(self._profiles.keys())

    def get_profiles_by_category(self, category: str) -> Dict[str, RadarProfile]:
        """Get all profiles in a specific category."""
        return {
            name: profile
            for name, profile in self._profiles.items()
            if profile.category == category
        }

    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories."""
        categories = set(p.category for p in self._profiles.values())
        return sorted(list(categories))

    def add_profile(self, profile: RadarProfile, save: bool = True):
        """
        Add a new profile.

        Args:
            profile: The radar profile to add
            save: If True, save to user profiles directory
        """
        self._profiles[profile.name] = profile
        if save:
            self.save_profile(profile)

    def save_profile(self, profile: RadarProfile, filename: Optional[str] = None):
        """
        Save a profile to the user profiles directory.

        Args:
            profile: The profile to save
            filename: Optional custom filename (without extension)
        """
        if filename is None:
            # Create filename from profile name (sanitize for filesystem)
            filename = "".join(c if c.isalnum() or c in "._- " else "_"
                              for c in profile.name).strip()

        filepath = self._user_dir / f"{filename}.yaml"
        profile.save_to_yaml(str(filepath))

    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile. Only user profiles can be deleted.

        Args:
            name: Name of the profile to delete

        Returns:
            True if profile was deleted, False if not found or is a default
        """
        if name not in self._profiles:
            return False

        # Check if it's a default profile
        default_file = self._defaults_dir / f"{name}.yaml"
        if default_file.exists():
            print(f"Cannot delete default profile: {name}")
            return False

        # Find and delete the user profile file
        for yaml_file in self._user_dir.glob("*.yaml"):
            try:
                profile = RadarProfile.from_yaml_file(str(yaml_file))
                if profile.name == name:
                    yaml_file.unlink()
                    del self._profiles[name]
                    return True
            except Exception:
                continue

        return False

    def create_custom_profile(self,
                              name: str,
                              base_profile: Optional[str] = None,
                              **kwargs) -> RadarProfile:
        """
        Create a new custom profile, optionally based on an existing one.

        Args:
            name: Name for the new profile
            base_profile: Name of profile to copy from (optional)
            **kwargs: Profile parameters to override

        Returns:
            The newly created profile
        """
        if base_profile and base_profile in self._profiles:
            # Copy from existing profile
            base = self._profiles[base_profile]
            data = base.to_dict()
            data['name'] = name
            data['category'] = 'Custom'
            data.update(kwargs)
            profile = RadarProfile.from_dict(data)
        else:
            # Create from scratch
            profile = RadarProfile(name=name, category='Custom', **kwargs)

        return profile

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, name: str) -> bool:
        return name in self._profiles

    def __iter__(self):
        return iter(self._profiles.values())


# Convenience function to get a default profile manager
_default_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """Get the default profile manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ProfileManager()
    return _default_manager
