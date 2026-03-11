"""
Тесты для модуля entity_extractor
"""

import pytest
from src.entity_extractor import (
    EntityExtractor,
    RegexExtractor,
    EntityType,
    Entity
)


class TestRegexExtractor:
    """Тесты для regex экстрактора"""
    
    def test_extract_email(self):
        """Проверка извлечения email"""
        extractor = RegexExtractor()
        entities = extractor.extract("Пишите на test@example.com или support@mail.ru")
        
        emails = [e for e in entities if e.entity_type == EntityType.EMAIL]
        assert len(emails) == 2
        assert "test@example.com" in [e.text for e in emails]
        assert "support@mail.ru" in [e.text for e in emails]
    
    def test_extract_url(self):
        """Проверка извлечения URL"""
        extractor = RegexExtractor()
        entities = extractor.extract("Сайт: https://github.com и http://example.com/page")
        
        urls = [e for e in entities if e.entity_type == EntityType.URL]
        assert len(urls) == 2
    
    def test_extract_phone(self):
        """Проверка извлечения телефона"""
        extractor = RegexExtractor()
        entities = extractor.extract("Звоните: +7 (999) 123-45-67 или 8-495-123-45-67")
        
        phones = [e for e in entities if e.entity_type == EntityType.PHONE]
        assert len(phones) == 2
    
    def test_extract_date(self):
        """Проверка извлечения даты"""
        extractor = RegexExtractor()
        entities = extractor.extract("Дата: 15.03.2024 и 1 января 2024")
        
        dates = [e for e in entities if e.entity_type == EntityType.DATE]
        assert len(dates) == 2
    
    def test_extract_technology(self):
        """Проверка извлечения технологий"""
        extractor = RegexExtractor()
        entities = extractor.extract("Используем Python и Django для разработки")
        
        techs = [e for e in entities if e.entity_type == EntityType.TECHNOLOGY]
        assert len(techs) >= 2
        tech_names = [e.text.lower() for e in techs]
        assert "python" in tech_names
        assert "django" in tech_names
    
    def test_extract_term(self):
        """Проверка извлечения терминов"""
        extractor = RegexExtractor()
        entities = extractor.extract("Машинное обучение и нейронные сети")
        
        terms = [e for e in entities if e.entity_type == EntityType.TERM]
        assert len(terms) >= 2


class TestEntityExtractor:
    """Тесты для комбинированного экстрактора"""
    
    def test_extractor_creation(self):
        """Проверка создания экстрактора"""
        extractor = EntityExtractor(use_ner=False)
        assert extractor is not None
        assert extractor.use_ner == False
    
    def test_extract_multiple_types(self):
        """Проверка извлечения нескольких типов"""
        extractor = EntityExtractor(use_ner=False)
        text = """
        Автор: Иванов И.И.
        Email: ivanov@university.ru
        Использовался Python и машинное обучение
        Дата: 15.03.2024
        Сайт: https://github.com/project
        """
        
        entities = extractor.extract(text, "test.txt", 0)
        
        # Должны быть найдены разные типы
        types_found = set(e.entity_type for e in entities)
        assert EntityType.EMAIL in types_found
        assert EntityType.TECHNOLOGY in types_found
        assert EntityType.DATE in types_found
        assert EntityType.URL in types_found
    
    def test_group_by_type(self):
        """Проверка группировки по типам"""
        extractor = EntityExtractor(use_ner=False)
        text = "Python и Django для ML. Пишите на test@example.com"
        
        entities = extractor.extract(text, "test.txt", 0)
        grouped = extractor.group_by_type(entities)
        
        assert EntityType.TECHNOLOGY in grouped
        assert EntityType.EMAIL in grouped
    
    def test_get_stats(self):
        """Проверка статистики"""
        extractor = EntityExtractor(use_ner=False)
        text = "Python использует Python для Python. Email: test@test.com"
        
        entities = extractor.extract(text, "test.txt", 0)
        stats = extractor.get_stats(entities)
        
        assert "Технология" in stats
        assert stats["Технология"] >= 1
        assert "Email" in stats


class TestEntityType:
    """Тесты для EntityType"""
    
    def test_entity_types(self):
        """Проверка типов сущностей"""
        assert EntityType.PERSON.value == "ФИО"
        assert EntityType.TECHNOLOGY.value == "Технология"
        assert EntityType.EMAIL.value == "Email"
        assert EntityType.PHONE.value == "Телефон"
        assert EntityType.URL.value == "URL"
        assert EntityType.DATE.value == "Дата"
        assert EntityType.ORGANIZATION.value == "Организация"
        assert EntityType.TERM.value == "Термин"
