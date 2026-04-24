"""
Preference Extractor for Gaprio.

Extracts user preferences from natural language messages using pattern matching.
Detected preferences are categorized and can be saved to USER.md for personalization.
"""

import re
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPreference:
    """A preference extracted from a user message."""
    category: str  # e.g., "name", "timezone", "response_style"
    value: str     # The actual preference value
    raw_match: str # The original matched text
    confidence: float = 1.0  # How confident we are (0-1)


class PreferenceExtractor:
    """
    Extracts user preferences from natural language.
    
    Uses pattern matching to detect common preference expressions
    and categorizes them for storage in USER.md.
    
    Example:
        extractor = PreferenceExtractor()
        prefs = extractor.extract_preferences("My name is Hanu")
        # Returns: [ExtractedPreference(category="name", value="Hanu", ...)]
    """
    
    # Patterns for different preference categories
    # Format: (pattern, category, value_group_index)
    PATTERNS: list[tuple[str, str, int]] = [
        # Name
        (r"(?:my name is|i'm|i am|call me) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "name", 1),
        
        # Role/Job
        (r"(?:my role is|i'm a|i am a|i work as|my job is) ([A-Za-z\s]+)", "role", 1),
        (r"i'm the ([A-Za-z\s]+) (?:at|for|of)", "role", 1),
        
        # Email
        (r"(?:my email is|email me at|reach me at) ([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", "email", 1),
        
        # Timezone
        (r"(?:my timezone is|i'm in|i am in) ([A-Z]{2,4}|[A-Za-z]+/[A-Za-z_]+)", "timezone", 1),
        (r"timezone[:\s]+([A-Z]{2,4}|[A-Za-z]+/[A-Za-z_]+)", "timezone", 1),
        
        # Response style preferences
        (r"i (?:prefer|like|want) (?:short|brief|concise) (?:responses?|summaries?|answers?)", "response_style", 0),
        (r"i (?:prefer|like|want) (?:detailed|long|thorough) (?:responses?|summaries?|answers?)", "response_style", 0),
        (r"(?:give me |use |prefer )bullet[- ]?points?", "response_style", 0),
        (r"(?:always |please )?(?:be |keep it )(?:brief|concise|short)", "response_style", 0),
        
        # Communication preferences  
        (r"always (.+?) (?:before|when|if)", "behavior", 1),
        (r"never (.+?) (?:without|unless)", "behavior", 1),
        (r"please (?:always|never) (.+)", "behavior", 1),
        (r"don't (?:ever )?(.+?) (?:without|unless) (?:asking|my permission)", "behavior", 1),
        
        # Remember statements - important facts
        (r"remember that (.+)", "notes", 1),
        (r"remember this[:\s]+(.+)", "notes", 1),
        (r"note that (.+)", "notes", 1),
        (r"keep in mind[:\s]+(.+)", "notes", 1),
        
        # Work preferences
        (r"i (?:work on|focus on|care about) (.+)", "work_focus", 1),
        (r"(?:my|our) (?:team|project|company) (?:is called |is |name is )(.+)", "team_info", 1),
        
        # Channel preferences
        (r"(?:my main|my primary|i mainly use) channel(?:s)? (?:is|are) (#?\w+(?:,?\s*#?\w+)*)", "primary_channels", 1),
    ]
    
    # Value normalizers for certain categories
    RESPONSE_STYLE_MAP = {
        "short": "Concise and brief",
        "brief": "Concise and brief", 
        "concise": "Concise and brief",
        "detailed": "Detailed and thorough",
        "long": "Detailed and thorough",
        "thorough": "Detailed and thorough",
        "bullet": "Use bullet points",
    }
    
    def extract_preferences(self, message: str) -> list[ExtractedPreference]:
        """
        Extract all preferences from a user message.
        
        Args:
            message: The user's message text
            
        Returns:
            List of extracted preferences
        """
        preferences = []
        message_lower = message.lower()
        
        for pattern, category, value_group in self.PATTERNS:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            
            for match in matches:
                try:
                    if value_group == 0:
                        # Use the whole match as the value
                        raw_value = match.group(0)
                    else:
                        raw_value = match.group(value_group)
                    
                    # Normalize the value
                    value = self._normalize_value(raw_value, category)
                    
                    pref = ExtractedPreference(
                        category=category,
                        value=value,
                        raw_match=match.group(0),
                    )
                    preferences.append(pref)
                    
                    logger.info(f"Extracted preference: {category} = {value}")
                    
                except (IndexError, AttributeError) as e:
                    logger.debug(f"Pattern match failed for {pattern}: {e}")
                    continue
        
        # Remove duplicates (same category)
        seen_categories = set()
        unique_prefs = []
        for pref in preferences:
            if pref.category not in seen_categories:
                seen_categories.add(pref.category)
                unique_prefs.append(pref)
        
        return unique_prefs
    
    def _normalize_value(self, value: str, category: str) -> str:
        """Normalize extracted values to consistent formats."""
        value = value.strip()
        
        if category == "name":
            # Capitalize name properly
            return value.title()
        
        elif category == "response_style":
            # Map to standard response style descriptions
            for key, mapped_value in self.RESPONSE_STYLE_MAP.items():
                if key in value.lower():
                    return mapped_value
            return value.capitalize()
        
        elif category == "timezone":
            # Keep timezone as-is but strip whitespace
            return value.upper() if len(value) <= 4 else value
        
        elif category == "primary_channels":
            # Clean up channel names
            channels = re.findall(r'#?(\w+)', value)
            return ", ".join(f"#{c}" for c in channels)
        
        return value.strip()
    
    def has_preference_indicators(self, message: str) -> bool:
        """
        Quick check if a message might contain preferences.
        
        Use this for early filtering before running full extraction.
        """
        indicators = [
            # Name/identity
            "my name", "i'm ", "i am ", "call me", "known as",
            # Preferences
            "i prefer", "i like", "i want", "i need",
            # Behavior requests
            "always", "never", "please", "don't ",
            # Context info
            "timezone", "my team", "my project", "my company",
            # Style preferences
            "bullet point", "brief", "concise", "detailed", "short", "long",
            # Memory/learning cues
            "remember that", "remember this", "know that", "note that",
            "keep in mind", "for future", "from now on",
            # About user
            "about me", "i work", "i focus", "my role", "my job",
        ]
        message_lower = message.lower()
        return any(ind in message_lower for ind in indicators)
