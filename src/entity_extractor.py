"""
Модуль извлечения сущностей (NER + Regex).

Что извлекает:
- ФИО (фамилии, имена, отчества)
- Технологии (языки программирования, фреймворки, библиотеки)
- Даты
- Email
- Телефоны
- URL/ссылки
- Организации
- Ключевые термины
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    """Типы сущностей."""
    PERSON = "ФИО"
    TECHNOLOGY = "Технология"
    DATE = "Дата"
    EMAIL = "Email"
    PHONE = "Телефон"
    URL = "URL"
    ORGANIZATION = "Организация"
    TERM = "Термин"


@dataclass
class Entity:
    """Сущность с метаданными."""
    text: str
    entity_type: EntityType
    source_file: str
    chunk_index: int
    confidence: float  # Уверенность (0.0 - 1.0)
    start_pos: int = 0
    end_pos: int = 0


class RegexExtractor:
    """Извлечение сущностей через регулярные выражения."""
    
    # Паттерны для извлечения
    PATTERNS = {
        EntityType.EMAIL: re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        EntityType.URL: re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
        EntityType.PHONE: re.compile(r'(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'),
        EntityType.DATE: re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\b'),
    }
    
    # Технологии (список можно расширять)
    TECHNOLOGIES = [
        # Языки программирования
        'Python', 'Java', 'C++', 'C#', 'JavaScript', 'TypeScript', 'Go', 'Rust',
        'Ruby', 'PHP', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB',
        
        # Фреймворки
        'Django', 'Flask', 'FastAPI', 'Spring', 'React', 'Vue', 'Angular',
        'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn', 'Pandas', 'NumPy',
        
        # Библиотеки
        'NumPy', 'Pandas', 'Matplotlib', 'Seaborn', 'OpenCV',
        
        # Технологии
        'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Git', 'Linux',
        'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch',
        
        # ML/AI
        'BERT', 'GPT', 'Transformer', 'NER', 'NLP', 'LSTM', 'CNN', 'RNN',
        'mBART', 'LaBSE', 'FAISS', 'XLM-RoBERTa',
    ]
    
    # Ключевые термины для студенческих отчётов
    TERMS = [
        'машинное обучение', 'глубокое обучение', 'нейронные сети',
        'обработка естественного языка', 'компьютерное зрение',
        'извлечение сущностей', 'семантический поиск', 'векторное представление',
        'трансформер', 'attention mechanism', 'fine-tuning',
        'курсовая работа', 'дипломный проект', 'ВКР', 'отчёт по практике',
        'научный руководитель', 'преподаватель', 'студент',
    ]
    
    def extract(self, text: str, source_file: str = "", chunk_index: int = 0) -> List[Entity]:
        """Извлечь сущности из текста."""
        entities = []
        
        # Извлекаем по regex паттернам
        for entity_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    confidence=0.95,  # Высокая уверенность для regex
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
        
        # Ищем технологии
        for tech in self.TECHNOLOGIES:
            for match in re.finditer(r'\b' + re.escape(tech) + r'\b', text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    entity_type=EntityType.TECHNOLOGY,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
        
        # Ищем термины
        for term in self.TERMS:
            for match in re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    entity_type=EntityType.TERM,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    confidence=0.80,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
        
        return entities


class NERExtractor:
    """NER модель для извлечения сущностей."""
    
    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-ner"):
        """
        Инициализация NER экстрактора.
        
        Args:
            model_name: Название модели для NER (русская по умолчанию)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Ленивая загрузка модели."""
        if self.model is None:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch
            
            print(f"Загрузка NER модели: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            print("NER модель загружена")
    
    def extract(self, text: str, source_file: str = "", chunk_index: int = 0) -> List[Entity]:
        """Извлечь сущности с помощью NER модели."""
        # Пока заглушка - NER модель тяжёлая, можно не использовать
        # Если нужна реальная NER - раскомментировать код ниже
        
        # from transformers import pipeline
        # self._load_model()
        # 
        # ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        # ner_results = ner_pipeline(text)
        # 
        # entities = []
        # for result in ner_results:
        #     entity_type = self._map_ner_type(result['entity'])
        #     if entity_type:
        #         entity = Entity(
        #             text=result['word'],
        #             entity_type=entity_type,
        #             source_file=source_file,
        #             chunk_index=chunk_index,
        #             confidence=result['score'],
        #         )
        #         entities.append(entity)
        # 
        # return entities
        
        return []  # Возвращаем пустой список (NER отключена по умолчанию)
    
    def _map_ner_type(self, ner_tag: str) -> Optional[EntityType]:
        """Сопоставить NER тег с EntityType."""
        mapping = {
            'PER': EntityType.PERSON,
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'ORGANIZATION': EntityType.ORGANIZATION,
            'LOC': EntityType.ORGANIZATION,  # Локации мапим на организации
            'MISC': EntityType.TERM,
        }
        return mapping.get(ner_tag.replace('B-', '').replace('I-', ''))


class EntityExtractor:
    """
    Комбинированный экстрактор сущностей.
    
    Использует:
    1. Regex для точных паттернов (email, телефоны, URL)
    2. Словарь для технологий и терминов
    3. NER модель (опционально) для ФИО и организаций
    """
    
    def __init__(self, use_ner: bool = False):
        """
        Инициализация экстрактора.
        
        Args:
            use_ner: Использовать ли NER модель (медленнее, но находит ФИО)
        """
        self.regex_extractor = RegexExtractor()
        self.ner_extractor = NERExtractor() if use_ner else None
        self.use_ner = use_ner
    
    def extract(self, text: str, source_file: str = "", chunk_index: int = 0) -> List[Entity]:
        """Извлечь все сущности из текста."""
        entities = []
        
        # 1. Regex + словарь
        regex_entities = self.regex_extractor.extract(text, source_file, chunk_index)
        entities.extend(regex_entities)
        
        # 2. NER (если включена)
        if self.use_ner and self.ner_extractor:
            ner_entities = self.ner_extractor.extract(text, source_file, chunk_index)
            entities.extend(ner_entities)
        
        # Удаляем дубликаты (одинаковый текст в одной позиции)
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.text.lower(), entity.start_pos, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_from_chunks(self, chunks) -> List[Entity]:
        """Извлечь сущности из списка чанков."""
        all_entities = []
        
        for chunk in chunks:
            entities = self.extract(
                text=chunk.text,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index
            )
            all_entities.extend(entities)
        
        return all_entities
    
    def group_by_type(self, entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
        """Сгруппировать сущности по типам."""
        grouped = {}
        for entity in entities:
            if entity.entity_type not in grouped:
                grouped[entity.entity_type] = []
            grouped[entity.entity_type].append(entity)
        return grouped
    
    def get_stats(self, entities: List[Entity]) -> Dict[str, int]:
        """Получить статистику по сущностям."""
        stats = {}
        for entity in entities:
            type_name = entity.entity_type.value
            stats[type_name] = stats.get(type_name, 0) + 1
        return stats
