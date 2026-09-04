"""Theme tokens (single source of truth) for YoloForge visual design.

Design tokens decouple colors / radii / typography from widget styles so the
platform can be re-themed (token value swap) without touching every QSS rule.
Current QSS still carries literal colors; new components and the token system
migrate toward these constants incrementally (see docs/theme-system.md).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color tokens (Dark theme, canonical palette)
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    # Foundations
    'bg_base': '#060D14',
    'bg_panel': '#08101A',
    'bg_panel_alt': '#0C1722',
    'bg_elevated': '#101E2B',
    'bg_input': '#0B141F',
    'bg_hover': '#152334',
    'bg_selected': '#1A2C3F',

    # Borders
    'border_soft': 'rgba(111, 151, 184, 62)',
    'border_mid': 'rgba(137, 187, 214, 110)',
    'border_focus': '#36B7FF',

    # Text
    'text_primary': '#E7F5FC',
    'text_secondary': '#A9C6D8',
    'text_muted': '#6E8C9E',
    'text_on_accent': '#06131F',

    # Brand accents
    'accent_blue': '#36B7FF',
    'accent_cyan': '#62E8FF',
    'accent_green': '#45D483',
    'accent_amber': '#F5A524',
    'accent_red': '#FF6B6B',
    'accent_violet': '#B88CFF',
    'accent_orange': '#F28A2E',

    # Semantic (status)
    'status_success': '#45D483',
    'status_warning': '#F5A524',
    'status_error': '#FF6B6B',
    'status_info': '#62E8FF',
    'status_neutral': '#91A8B8',

    # Layers
    'overlay': 'rgba(8, 16, 25, 224)',

    # Chart series
    'series_0': '#36B7FF',
    'series_1': '#45D483',
    'series_2': '#F5A524',
    'series_3': '#FF6677',
    'series_4': '#B88CFF',
    'series_5': '#62E8FF',
    'series_6': '#F28AC8',
    'series_7': '#A4D65E',
}

# ---------------------------------------------------------------------------
# Spacing / radius / typography tokens (8px baseline)
# ---------------------------------------------------------------------------

SPACING: dict[str, int] = {
    'xs': 4,
    'sm': 8,
    'md': 12,
    'lg': 16,
    'xl': 24,
    'xxl': 32,
}

RADIUS: dict[str, int] = {
    'chip': 10,
    'card': 14,
    'panel': 16,
    'input': 8,
}

# Font-size ladder (px) — '有序的专业密度' gradient
TYPOGRAPHY: dict[str, int] = {
    'micro': 9,
    'caption': 10,
    'body': 12,
    'body_strong': 13,
    'section': 15,
    'page': 17,
    'display': 22,
}


def color(name: str, fallback: str = '#000000') -> str:
    """Resolve a color token by name."""
    return COLORS.get(name, fallback)


def series_color(index: int) -> str:
    """Cycle chart series colors."""
    return COLORS.get(f'series_{index % 8}', COLORS['series_0'])
