"""
Модуль предобработки текста:
- Очистка от лишних пробелов и служебных символов
- Нормализация текста
- Разбиение на фрагменты (чанки) по 500–1000 символов
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """
    Фрагмент текста с метаданными.
    
    Attributes:
        text: Текст фрагмента
        source_file: Путь к исходному файлу
        chunk_index: Номер фрагмента в документе
        start_pos: Позиция начала в исходном тексте
        end_pos: Позиция конца в исходном тексте
    """
    text: str
    source_file: str
    chunk_index: int
    start_pos: int
    end_pos: int


class TextPreprocessor:
    """
    Предобработка текста: очистка и нормализация.
    """
    
    # Паттерны для очистки
    MULTIPLE_SPACES = re.compile(r'\s+')
    SPECIAL_CHARS = re.compile(r'[^\w\s.,;:!?()\[\]{}"\'\-–—а-яА-Яa-zA-Z0-9]')
    
    def clean(self, text: str) -> str:
        """
        Очистить текст от лишнего мусора.
        
        Что делает:
        - Заменяет множественные пробелы/табы/переводы строк на один пробел
        - Удаляет служебные символы (кроме базовой пунктуации)
        - Обрезает пробелы по краям
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        # Заменяем множественные пробелы на один
        text = self.MULTIPLE_SPACES.sub(' ', text)
        
        # Удаляем служебные символы (оставляем базовую пунктуацию)
        text = self.SPECIAL_CHARS.sub('', text)
        
        # Обрезаем пробелы по краям
        text = text.strip()
        
        return text
    
    def normalize(self, text: str) -> str:
        """
        Нормализовать текст.
        
        Что делает:
        - Приводит к нижнему регистру (опционально)
        - Заменяет ё на е (для единообразия)
        
        Args:
            text: Исходный текст
            
        Returns:
            Нормализованный текст
        """
        # Заменяем ё на е (распространено в русских текстах)
        text = text.replace('ё', 'е').replace('Ё', 'Е')
        
        return text


class TextChunker:
    """
    Разбиение текста на фрагменты (чанки).
    
    Почему это важно:
    - Семантический поиск работает по фрагментам
    - Большие документы разбиваются на части по 500–1000 символов
    - Каждый фрагмент будет иметь свой эмбеддинг
    """
    
    def __init__(self, chunk_size: int = 750, overlap: int = 100):
        """
        Инициализация чанкера.
        
        Args:
            chunk_size: Размер одного фрагмента в символах (по умолчанию 750)
            overlap: Перекрытие между фрагментами в символах (по умолчанию 100)
                     Нужно для сохранения контекста между соседними чанками
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source_file: str = "") -> List[TextChunk]:
        """
        Разбить текст на фрагменты.
        
        Алгоритм:
        1. Разбиваем текст на предложения (по .!?;)
        2. Собираем предложения в чанки нужного размера
        3. Делаем перекрытие между чанками для сохранения контекста
        
        Args:
            text: Текст для разбиения
            source_file: Путь к исходному файлу (для метаданных)
            
        Returns:
            Список фрагментов с метаданными
        """
        if not text.strip():
            return []
        
        chunks = []
        chunk_index = 0
        
        # Разбиваем на предложения
        sentences = re.split(r'(?<=[.!?;])\s+', text)
        
        current_chunk = ""
        current_start = 0
        sentence_positions = []  # Позиции начала предложений
        
        pos = 0
        for i, sentence in enumerate(sentences):
            sentence_positions.append(pos)
            pos += len(sentence) + 1  # +1 за пробел
        
        for i, sentence in enumerate(sentences):
            # Если предложение очень длинное (больше chunk_size), разбиваем его
            if len(sentence) > self.chunk_size:
                # Если есть накопленный чанк — сохраняем его
                if current_chunk.strip():
                    chunk = TextChunk(
                        text=current_chunk.strip(),
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Разбиваем длинное предложение на части
                for j in range(0, len(sentence), self.chunk_size - self.overlap):
                    part = sentence[j:j + self.chunk_size]
                    start = sentence_positions[i] + j if i < len(sentence_positions) else 0
                    
                    chunk = TextChunk(
                        text=part.strip(),
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=start,
                        end_pos=start + len(part)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = ""
                current_start = sentence_positions[i] + len(sentence) + 1 if i < len(sentence_positions) else 0
                continue
            
            # Добавляем предложение к текущему чанку
            if not current_chunk:
                current_chunk = sentence
                current_start = sentence_positions[i] if i < len(sentence_positions) else 0
            else:
                current_chunk += " " + sentence
            
            # Если чанк набрал нужный размер — сохраняем
            if len(current_chunk) >= self.chunk_size:
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk)
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Оставляем перекрытие для следующего чанка
                # Важно: сохраняем позицию начала overlap для следующего чанка
                overlap_start = current_start + len(current_chunk) - self.overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else ""
                current_chunk = overlap_text
                current_start = overlap_start
        
        # Сохраняем последний чанк
        if current_chunk.strip():
            chunk = TextChunk(
                text=current_chunk.strip(),
                source_file=source_file,
                chunk_index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks


class DocumentProcessor:
    """
    Обработчик документов: объединяет предобработку и разбиение на чанки.
    
    Использование:
        processor = DocumentProcessor()
        chunks = processor.process("текст документа", "file.docx")
    """
    
    def __init__(self, chunk_size: int = 750, overlap: int = 100):
        """
        Инициализация процессора.
        
        Args:
            chunk_size: Размер чанка в символах
            overlap: Перекрытие между чанками
        """
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker(chunk_size, overlap)
    
    def process(self, text: str, source_file: str = "") -> List[TextChunk]:
        """
        Обработать документ: очистить и разбить на фрагменты.
        
        Этапы:
        1. Очистка текста (удаление мусора)
        2. Нормализация (замена ё → е)
        3. Разбиение на чанки
        
        Args:
            text: Исходный текст документа
            source_file: Путь к исходному файлу
            
        Returns:
            Список фрагментов с метаданными
        """
        # Очистка
        cleaned_text = self.preprocessor.clean(text)
        
        # Нормализация
        normalized_text = self.preprocessor.normalize(cleaned_text)
        
        # Разбиение на чанки
        chunks = self.chunker.chunk(normalized_text, source_file)
        
        return chunks
    
    def process_multiple(self, documents: dict[str, str], progress_callback=None) -> List[TextChunk]:
        """
        Обработать несколько документов.
        
        Args:
            documents: Словарь {путь_к_файлу: текст}
            progress_callback: Callback для прогресса (current, total, filename)
            
        Returns:
            Список всех фрагментов из всех документов
        """
        all_chunks = []
        total = len(documents)
        
        for i, (file_path, text) in enumerate(documents.items()):
            chunks = self.process(text, file_path)
            all_chunks.extend(chunks)
            
            if progress_callback:
                progress_callback(i + 1, total, file_path)
        
        return all_chunks
