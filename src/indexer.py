"""
Модуль семантической индексации документов.

Что делает:
- Создаёт эмбеддинги для чанков (sentence-transformers)
- Сохраняет в FAISS-индекс для быстрого поиска
- Сохраняет метаданные для связи с чанками
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.preprocessor import TextChunk


@dataclass
class IndexedChunk:
    """Чанк с эмбеддингом и метаданными."""
    chunk_id: int
    text: str
    source_file: str
    chunk_index: int
    start_pos: int
    end_pos: int
    embedding: np.ndarray


class DocumentIndexer:
    """
    Индексатор документов.
    
    Создаёт FAISS-индекс для семантического поиска по чанкам.
    
    Использование:
        indexer = DocumentIndexer()
        indexer.build_index(chunks)
        indexer.save("index.faiss")
    """
    
    # Модель для создания эмбеддингов (русская + английская, поменьше)
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, model_name: str = None):
        """
        Инициализация индексатора.
        
        Args:
            model_name: Название модели sentence-transformers
        """
        self.model_name = model_name or self.MODEL_NAME
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[IndexedChunk] = []
        self.dimension: int = 768  # Размер эмбеддинга для LaBSE
    
    @property
    def is_built(self) -> bool:
        """Построен ли индекс."""
        return self.index is not None and len(self.chunks) > 0
    
    def _load_model(self):
        """Ленивая загрузка модели."""
        if self.model is None:
            print(f"Загрузка модели: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"Размер эмбеддинга: {self.dimension}")
    
    def build_index(self, chunks: List[TextChunk], batch_size: int = 32):
        """
        Построить индекс для списка чанков.
        
        Args:
            chunks: Список чанков для индексации
            batch_size: Размер батча для создания эмбеддингов
        """
        if not chunks:
            raise ValueError("Список чанков пуст!")
        
        print(f"Построение индекса для {len(chunks)} чанков...")
        
        # Загружаем модель
        self._load_model()
        
        # Создаём эмбеддинги
        texts = [chunk.text for chunk in chunks]
        print("Создание эмбеддингов...")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 нормализация для косинусного сходства
            )
            embeddings.append(batch_embeddings)
            print(f"  Обработано {min(i + batch_size, len(texts))}/{len(texts)} чанков")
        
        # Объединяем все эмбеддинги
        all_embeddings = np.vstack(embeddings).astype('float32')
        
        # Создаём FAISS индекс (Inner Product = косинусное сходство для нормализованных векторов)
        print("Создание FAISS индекса...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(all_embeddings)
        
        # Сохраняем чанки с эмбеддингами
        self.chunks = []
        for i, chunk in enumerate(chunks):
            indexed_chunk = IndexedChunk(
                chunk_id=i,
                text=chunk.text,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                embedding=all_embeddings[i]
            )
            self.chunks.append(indexed_chunk)
        
        print(f"Индекс построен! {len(self.chunks)} чанков, размер индекса: {self.index.ntotal}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[IndexedChunk, float]]:
        """
        Поиск релевантных чанков по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            
        Returns:
            Список кортежей (чанк, score) отсортированных по релевантности
        """
        if not self.is_built:
            raise RuntimeError("Индекс не построен! Вызовите build_index() сначала.")
        
        # Загружаем модель если нужно
        self._load_model()
        
        # Создаём эмбеддинг запроса
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Ограничиваем top_k количеством чанков в индексе
        actual_top_k = min(top_k, len(self.chunks))
        
        # Ищем в индексе
        scores, indices = self.index.search(query_embedding, actual_top_k)
        
        # Возвращаем результаты
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Проверяем валидность индекса
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """
        Сохранить индекс и метаданные.
        
        Args:
            path: Путь для сохранения (без расширения)
        """
        if not self.is_built:
            raise RuntimeError("Нечего сохранять! Индекс не построен.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем FAISS индекс
        faiss.write_index(self.index, str(path) + ".faiss")
        
        # Сохраняем чанки и метаданные
        with open(str(path) + ".pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'model_name': self.model_name,
                'dimension': self.dimension
            }, f)
        
        print(f"Индекс сохранён: {path}.faiss, {path}.pkl")
    
    def load(self, path: str):
        """
        Загрузить индекс из файлов.
        
        Args:
            path: Путь к файлам индекса (без расширения)
        """
        path = Path(path)
        
        # Загружаем метаданные
        with open(str(path) + ".pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.model_name = data['model_name']
            self.dimension = data['dimension']
        
        # Загружаем FAISS индекс
        self.index = faiss.read_index(str(path) + ".faiss")
        
        # Загружаем модель
        self._load_model()
        
        print(f"Индекс загружен: {len(self.chunks)} чанков")
    
    def get_stats(self) -> dict:
        """Получить статистику индекса."""
        if not self.is_built:
            return {}
        
        # Считаем чанки по файлам
        files_count = {}
        for chunk in self.chunks:
            files_count[chunk.source_file] = files_count.get(chunk.source_file, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'total_files': len(files_count),
            'files': files_count,
            'dimension': self.dimension,
            'model': self.model_name
        }
