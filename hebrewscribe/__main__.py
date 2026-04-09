"""Allow running as: python -m hebrewscribe"""

from hebrewscribe.utils import setup_logging

setup_logging()

from hebrewscribe.app import main

main()
