"""
DocInsight - Система анализа корпуса документов
Точка входа приложения
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.main_window import main

if __name__ == "__main__":
    main()
