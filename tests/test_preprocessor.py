"""
Тесты для модуля preprocessor
"""

import pytest
from src.preprocessor import TextPreprocessor, TextChunker, DocumentProcessor, TextChunk


class TestTextPreprocessor:
    """Тесты для очистителя текста"""
    
    def test_preprocessor_creation(self):
        """Проверка создания препроцессора"""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
    
    def test_clean_multiple_spaces(self):
        """Проверка удаления множественных пробелов"""
        preprocessor = TextPreprocessor()
        text = "Привет   \t\n  мир"
        cleaned = preprocessor.clean(text)
        assert cleaned == "Привет мир"
    
    def test_clean_special_chars(self):
        """Проверка удаления служебных символов"""
        preprocessor = TextPreprocessor()
        text = "Текст с #символами* и %знаками^"
        cleaned = preprocessor.clean(text)
        assert "#" not in cleaned
        assert "*" not in cleaned
        assert "%" not in cleaned
    
    def test_clean_strip(self):
        """Проверка обрезки пробелов по краям"""
        preprocessor = TextPreprocessor()
        text = "   текст по краям   "
        cleaned = preprocessor.clean(text)
        assert cleaned == "текст по краям"
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
    
    def test_normalize_yo(self):
        """Проверка замены ё на е"""
        preprocessor = TextPreprocessor()
        text = "Ёжик и ёлка"
        normalized = preprocessor.normalize(text)
        assert "ё" not in normalized
        assert "е" in normalized


class TestTextChunker:
    """Тесты для разбиения на чанки"""
    
    def test_chunker_creation(self):
        """Проверка создания чанкера"""
        chunker = TextChunker(chunk_size=500, overlap=50)
        assert chunker is not None
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
    
    def test_chunk_empty_text(self):
        """Проверка обработки пустого текста"""
        chunker = TextChunker()
        chunks = chunker.chunk("", "test.txt")
        assert len(chunks) == 0
    
    def test_chunk_short_text(self):
        """Проверка обработки короткого текста"""
        chunker = TextChunker(chunk_size=100)
        text = "Один короткий предложение."
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Один короткий предложение."
    
    def test_chunk_long_text(self):
        """Проверка разбиения длинного текста"""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "Первое предложение. Второе предложение. Третье предложение. " * 10
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) > 1
        
        # Проверяем, что чанки не пустые
        for chunk in chunks:
            assert len(chunk.text) > 0
    
    def test_chunk_metadata(self):
        """Проверка метаданных чанков"""
        chunker = TextChunker(chunk_size=100)
        text = "Первое предложение. Второе предложение."
        chunks = chunker.chunk(text, "document.docx")
        
        assert len(chunks) > 0
        assert chunks[0].source_file == "document.docx"
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_pos >= 0
        assert chunks[0].end_pos > chunks[0].start_pos


class TestTextChunk:
    """Тесты для класса TextChunk"""
    
    def test_chunk_creation(self):
        """Проверка создания чанка"""
        chunk = TextChunk(
            text="Тестовый текст",
            source_file="test.docx",
            chunk_index=0,
            start_pos=0,
            end_pos=14
        )
        assert chunk.text == "Тестовый текст"
        assert chunk.source_file == "test.docx"
        assert chunk.chunk_index == 0


class TestDocumentProcessor:
    """Тесты для процессора документов"""
    
    def test_processor_creation(self):
        """Проверка создания процессора"""
        processor = DocumentProcessor()
        assert processor is not None
    
    def test_process_full_pipeline(self):
        """Проверка полного цикла обработки"""
        processor = DocumentProcessor(chunk_size=100)
        text = "  Привет   мир!  Это тестовый   документ.  "
        chunks = processor.process(text, "test.docx")
        
        assert len(chunks) > 0
        # Проверяем, что текст очищен (нет множественных пробелов)
        for chunk in chunks:
            assert "  " not in chunk.text
    
    def test_process_multiple_documents(self):
        """Проверка обработки нескольких документов"""
        processor = DocumentProcessor()
        documents = {
            "doc1.txt": "Первый документ. Текст здесь.",
            "doc2.txt": "Второй документ. Другой текст.",
        }
        chunks = processor.process_multiple(documents)
        
        assert len(chunks) > 0
        
        # Проверяем, что чанки из разных документов
        source_files = {chunk.source_file for chunk in chunks}
        assert "doc1.txt" in source_files
        assert "doc2.txt" in source_files
