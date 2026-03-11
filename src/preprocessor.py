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
    Разбивает ТОЛЬКО по границам предложений — никаких обрывов!
    """

    def __init__(self, chunk_size: int = 750, overlap: int = 100):
        """
        Args:
            chunk_size: Целевой размер чанка (по умолчанию 750)
            overlap: Перекрытие в символах (по умолчанию 100)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source_file: str = "") -> List[TextChunk]:
        """Разбить текст на фрагменты ТОЛЬКО по границам предложений."""
        if not text.strip():
            return []

        chunks = []
        chunk_index = 0
        
        # Разбиваем на предложения
        sentences = re.split(r'(?<=[.!?;])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Вычисляем позиции предложений
        sentence_positions = []
        pos = 0
        for sentence in sentences:
            start = text.find(sentence, pos)
            sentence_positions.append(start)
            pos = start + len(sentence)
        
        current_sentences = []
        current_start = 0
        
        for i, sentence in enumerate(sentences):
            # Проверяем, влезет ли предложение в текущий чанк
            test_chunk = " ".join(current_sentences + [sentence])
            
            if len(test_chunk) <= self.chunk_size:
                # Влезает — добавляем
                if not current_sentences:
                    current_start = sentence_positions[i]
                current_sentences.append(sentence)
            else:
                # Не влезает — сохраняем текущий чанк
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunk = TextChunk(
                        text=chunk_text,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Определяем overlap — берём последние предложения, которые влезут в overlap
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current_sentences):
                        if overlap_len + len(s) + 1 <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break
                    
                    # Начинаем новый чанк с overlap + текущее предложение
                    current_sentences = overlap_sentences + [sentence] if overlap_sentences else [sentence]
                    # Находим позицию начала первого предложения overlap
                    if overlap_sentences:
                        current_start = sentence_positions[i - len(overlap_sentences)]
                    else:
                        current_start = sentence_positions[i]
                else:
                    # Текущий чанк пустой (предложение больше chunk_size)
                    # Разбиваем предложение по словам
                    words = sentence.split()
                    temp_chunk = []
                    temp_start = sentence_positions[i]
                    
                    for word in words:
                        if len(" ".join(temp_chunk + [word])) > self.chunk_size:
                            if temp_chunk:
                                chunk_text = " ".join(temp_chunk)
                                chunk = TextChunk(
                                    text=chunk_text,
                                    source_file=source_file,
                                    chunk_index=chunk_index,
                                    start_pos=temp_start,
                                    end_pos=temp_start + len(chunk_text)
                                )
                                chunks.append(chunk)
                                chunk_index += 1
                                temp_chunk = []
                        temp_chunk.append(word)
                    
                    if temp_chunk:
                        chunk_text = " ".join(temp_chunk)
                        chunk = TextChunk(
                            text=chunk_text,
                            source_file=source_file,
                            chunk_index=chunk_index,
                            start_pos=temp_start,
                            end_pos=temp_start + len(chunk_text)
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        current_sentences = []
                        current_start = sentence_positions[i] + len(sentence)
        
        # Сохраняем последний чанк
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = TextChunk(
                text=chunk_text,
                source_file=source_file,
                chunk_index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(chunk_text)
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
