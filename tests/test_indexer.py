"""
Тесты для модуля indexer
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.preprocessor import TextChunk, DocumentProcessor
from src.indexer import DocumentIndexer, IndexedChunk


class TestDocumentIndexer:
    """Тесты для индексатора документов"""
    
    def test_indexer_creation(self):
        """Проверка создания индексатора"""
        indexer = DocumentIndexer()
        assert indexer is not None
        assert not indexer.is_built
    
    def test_build_index(self):
        """Проверка построения индекса"""
        # Создаём тестовые чанки
        chunks = [
            TextChunk(text="Машинное обучение это интересно", source_file="test.txt", chunk_index=0, start_pos=0, end_pos=35),
            TextChunk(text="Нейронные сети используются везде", source_file="test.txt", chunk_index=1, start_pos=36, end_pos=70),
            TextChunk(text="Глубокое обучение требует данных", source_file="test.txt", chunk_index=2, start_pos=71, end_pos=105),
        ]
        
        indexer = DocumentIndexer()
        indexer.build_index(chunks)
        
        assert indexer.is_built
        assert len(indexer.chunks) == 3
        assert indexer.index.ntotal == 3
    
    def test_search(self):
        """Проверка поиска"""
        chunks = [
            TextChunk(text="Машинное обучение это интересно", source_file="test.txt", chunk_index=0, start_pos=0, end_pos=35),
            TextChunk(text="Нейронные сети используются везде", source_file="test.txt", chunk_index=1, start_pos=36, end_pos=70),
            TextChunk(text="Глубокое обучение требует данных", source_file="test.txt", chunk_index=2, start_pos=71, end_pos=105),
        ]
        
        indexer = DocumentIndexer()
        indexer.build_index(chunks)
        
        # Поиск
        results = indexer.search("машинное обучение", top_k=2)
        
        assert len(results) == 2
        # Первый результат должен быть про машинное обучение
        assert "машинное обучение" in results[0][0].text.lower() or "обучение" in results[0][0].text.lower()
    
    def test_save_and_load(self):
        """Проверка сохранения и загрузки"""
        chunks = [
            TextChunk(text="Тестовый текст 1", source_file="test.txt", chunk_index=0, start_pos=0, end_pos=16),
            TextChunk(text="Тестовый текст 2", source_file="test.txt", chunk_index=1, start_pos=17, end_pos=33),
        ]
        
        indexer = DocumentIndexer()
        indexer.build_index(chunks)
        
        # Сохраняем
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_index")
            indexer.save(index_path)
            
            # Проверяем, что файлы созданы
            assert os.path.exists(index_path + ".faiss")
            assert os.path.exists(index_path + ".pkl")
            
            # Загружаем
            new_indexer = DocumentIndexer()
            new_indexer.load(index_path)
            
            assert new_indexer.is_built
            assert len(new_indexer.chunks) == 2
    
    def test_get_stats(self):
        """Проверка статистики"""
        chunks = [
            TextChunk(text="Текст 1", source_file="file1.txt", chunk_index=0, start_pos=0, end_pos=7),
            TextChunk(text="Текст 2", source_file="file1.txt", chunk_index=1, start_pos=8, end_pos=15),
            TextChunk(text="Текст 3", source_file="file2.txt", chunk_index=0, start_pos=0, end_pos=7),
        ]
        
        indexer = DocumentIndexer()
        indexer.build_index(chunks)
        
        stats = indexer.get_stats()
        
        assert stats['total_chunks'] == 3
        assert stats['total_files'] == 2
        assert 'file1.txt' in stats['files']
        assert 'file2.txt' in stats['files']
        assert stats['files']['file1.txt'] == 2
        assert stats['files']['file2.txt'] == 1


class TestIndexedChunk:
    """Тесты для IndexedChunk"""
    
    def test_indexed_chunk_creation(self):
        """Проверка создания IndexedChunk"""
        import numpy as np
        
        chunk = IndexedChunk(
            chunk_id=0,
            text="Тест",
            source_file="test.txt",
            chunk_index=0,
            start_pos=0,
            end_pos=4,
            embedding=np.array([0.1, 0.2, 0.3])
        )
        
        assert chunk.chunk_id == 0
        assert chunk.text == "Тест"
        assert chunk.source_file == "test.txt"
        assert len(chunk.embedding) == 3
