#!/usr/bin/env python3
"""
Render SVG icons to multi-DPI PNG files for tkinter consumption.

This build-time script converts SVG icons from assets/icons/ to PNG files
at multiple scales (1x, 1.5x, 2x, 3x) in hebrewscribe/icons/.
"""

import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print(
        "Error: cairosvg is not installed.\n"
        "Install it with: pip install cairosvg\n"
        "On macOS, you may need to install cairo first: brew install cairo"
    )
    sys.exit(1)


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    svg_dir = project_root / "assets" / "icons"
    output_dir = project_root / "hebrewscribe" / "icons"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define scales: (suffix, size_px)
    scales = [
        ("_1x", 24),
        ("_1_5x", 36),
        ("_2x", 48),
        ("_3x", 72),
    ]

    # Collect results for summary
    rendered_icons = []

    # Render all SVG files
    svg_files = sorted(svg_dir.glob("*.svg"))

    if not svg_files:
        print(f"Warning: No SVG files found in {svg_dir}")
        return

    for svg_file in svg_files:
        icon_name = svg_file.stem

        for suffix, size_px in scales:
            output_file = output_dir / f"{icon_name}{suffix}.png"

            try:
                cairosvg.svg2png(
                    url=str(svg_file),
                    write_to=str(output_file),
                    output_width=size_px,
                    output_height=size_px,
                )
            except Exception as e:
                print(f"Error rendering {svg_file.name} at {size_px}px: {e}")
                sys.exit(1)

        rendered_icons.append(icon_name)

    # Print summary
    print(f"Successfully rendered {len(rendered_icons)} icons at 4 scales:")
    for icon in rendered_icons:
        print(f"  - {icon}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Scale files: _1x (24px), _1_5x (36px), _2x (48px), _3x (72px)")


if __name__ == "__main__":
    main()
