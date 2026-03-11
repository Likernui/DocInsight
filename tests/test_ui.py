"""
Тесты для модуля UI
"""

import pytest
from src.ui.main_window import MainWindow


class TestMainWindow:
    """Тесты для главного окна"""
    
    def test_window_creation(self, qtbot):
        """Проверка создания окна"""
        window = MainWindow()
        qtbot.addWidget(window)
        assert window is not None
    
    def test_window_title(self, qtbot):
        """Проверка заголовка окна"""
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.windowTitle() == "DocInsight - Система анализа документов"
    
    def test_window_minimum_size(self, qtbot):
        """Проверка минимального размера"""
        window = MainWindow()
        qtbot.addWidget(window)
        min_size = window.minimumSize()
        assert min_size.width() == 1000
        assert min_size.height() == 700
    
    def test_initial_state(self, qtbot):
        """Проверка начального состояния"""
        window = MainWindow()
        qtbot.addWidget(window)
        
        # Кнопки должны быть отключены
        assert not window.btn_extract.isEnabled()
        assert not window.btn_process.isEnabled()
        
        # Счётчик файлов
        assert window.lbl_file_count.text() == "Загружено файлов: 0"
        
        # Списки пустые
        assert window.file_list.count() == 0
        assert window.txt_source.toPlainText() == ""
        assert window.txt_chunks.toPlainText() == ""
