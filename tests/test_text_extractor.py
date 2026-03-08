"""
Тесты для модуля text_extractor
"""

import pytest
from pathlib import Path
from src.text_extractor import (
    DocxExtractor,
    PdfExtractor,
    ImageExtractor,
    DocumentLoader
)


class TestDocxExtractor:
    """Тесты для DOCX экстрактора"""
    
    def test_extractor_creation(self):
        """Проверка создания экстрактора"""
        extractor = DocxExtractor()
        assert extractor is not None


class TestPdfExtractor:
    """Тесты для PDF экстрактора"""
    
    def test_extractor_creation(self):
        """Проверка создания экстрактора"""
        extractor = PdfExtractor()
        assert extractor is not None


class TestImageExtractor:
    """Тесты для OCR экстрактора"""
    
    def test_extractor_creation(self):
        """Проверка создания экстрактора"""
        extractor = ImageExtractor(languages=['ru', 'en'])
        assert extractor is not None
        assert extractor.languages == ['ru', 'en']


class TestDocumentLoader:
    """Тесты для универсального загрузчика"""
    
    def test_loader_creation(self):
        """Проверка создания загрузчика"""
        loader = DocumentLoader()
        assert loader is not None
    
    def test_supported_extensions(self):
        """Проверка поддерживаемых расширений"""
        loader = DocumentLoader()
        assert '.docx' in loader.SUPPORTED_EXTENSIONS
        assert '.pdf' in loader.SUPPORTED_EXTENSIONS
        assert '.png' in loader.SUPPORTED_EXTENSIONS
        assert '.jpg' in loader.SUPPORTED_EXTENSIONS
        assert '.jpeg' in loader.SUPPORTED_EXTENSIONS
    
    def test_file_not_found(self):
        """Проверка обработки несуществующего файла"""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.docx")
    
    def test_unsupported_extension(self):
        """Проверка обработки неподдерживаемого расширения"""
        loader = DocumentLoader()
        with pytest.raises(ValueError):
            loader.load("file.txt")


class TestGetExtension:
    """Тесты для получения расширения"""
    
    def test_lowercase_extension(self):
        """Проверка приведения расширения к нижнему регистру"""
        ext = DocxExtractor.get_extension("FILE.DOCX")
        assert ext == '.docx'
    
    def test_pdf_extension(self):
        """Проверка получения расширения PDF"""
        ext = PdfExtractor.get_extension("document.pdf")
        assert ext == '.pdf'
    
    def test_png_extension(self):
        """Проверка получения расширения PNG"""
        ext = ImageExtractor.get_extension("scan.PNG")
        assert ext == '.png'
