"""
Главное окно приложения DocInsight
Базовый интерфейс для тестирования модулей
"""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QFileDialog,
    QLabel,
    QProgressBar,
    QTabWidget,
    QMessageBox,
    QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap

from src.text_extractor import DocumentLoader
from src.preprocessor import DocumentProcessor
from src.indexer import DocumentIndexer
from src.entity_extractor import EntityExtractor


class WorkerThread(QThread):
    """Поток для длительных операций (чтобы не блокировать UI)"""
    
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(object)  # результат
    error = pyqtSignal(str)  # ошибка
    
    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.operation(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        
        # Инициализация модулей
        self.loader = DocumentLoader()
        self.processor = DocumentProcessor(chunk_size=750, overlap=100)
        self.indexer = DocumentIndexer()
        self.entity_extractor = EntityExtractor(use_ner=False)  # NER отключена (медленная)
        
        # Данные
        self.file_paths: list[str] = []
        self.extracted_texts: dict[str, str] = {}
        self.all_chunks = []
        self.index_built = False
        self.all_entities = []
        
        # Настройка окна
        self._init_ui()
        self._setup_styles()
    
    def _init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("DocInsight - Система анализа документов")
        self.setMinimumSize(1000, 700)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Заголовок с логотипом
        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Логотип
        logo_path = Path(__file__).parent.parent.parent / "data" / "logo.png"
        if logo_path.exists():
            logo_label = QLabel()
            pixmap = QPixmap(str(logo_path)).scaled(
                100, 100,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(pixmap)
            logo_label.setFixedSize(100, 100)
            header_layout.addWidget(logo_label)
        
        # Название
        title = QLabel("DocInsight")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title)
        
        main_layout.addLayout(header_layout)
        
        # Разделитель
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Левая панель - загрузка файлов
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Кнопка загрузки
        self.btn_load = QPushButton("📁 Выбрать файлы")
        self.btn_load.setFixedHeight(50)
        self.btn_load.setStyleSheet("font-size: 15px;")
        self.btn_load.clicked.connect(self._load_files)
        left_layout.addWidget(self.btn_load)
        
        # Счётчик файлов
        self.lbl_file_count = QLabel("Загружено файлов: 0")
        self.lbl_file_count.setObjectName("file_count")
        self.lbl_file_count.setFont(QFont("Arial", 11))
        left_layout.addWidget(self.lbl_file_count)
        
        # Список файлов
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(False)
        left_layout.addWidget(self.file_list)
        
        # Прогресс бар
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)
        
        # Кнопки тестирования
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.btn_extract = QPushButton("🔍 Извлечь текст")
        self.btn_extract.setFixedHeight(45)
        self.btn_extract.clicked.connect(self._extract_text)
        self.btn_extract.setEnabled(False)
        btn_layout.addWidget(self.btn_extract)
        
        self.btn_process = QPushButton("⚙️ Обработать")
        self.btn_process.setFixedHeight(45)
        self.btn_process.clicked.connect(self._process_documents)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)
        
        left_layout.addLayout(btn_layout)
        
        # Кнопка построения индекса
        self.btn_index = QPushButton("📚 Построить индекс")
        self.btn_index.setFixedHeight(45)
        self.btn_index.clicked.connect(self._build_index)
        self.btn_index.setEnabled(False)
        left_layout.addWidget(self.btn_index)
        
        # Кнопка извлечения сущностей
        self.btn_entities = QPushButton("🏷️ Извлечь сущности")
        self.btn_entities.setFixedHeight(45)
        self.btn_entities.clicked.connect(self._extract_entities)
        self.btn_entities.setEnabled(False)
        left_layout.addWidget(self.btn_entities)
        
        # Правая панель - результаты
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Вкладки
        self.tabs = QTabWidget()
        
        # Вкладка 1: Исходный текст
        self.txt_source = QTextEdit()
        self.txt_source.setReadOnly(True)
        self.txt_source.setFont(QFont("Menlo", 10))  # Menlo — аналог Consolas для macOS
        self.tabs.addTab(self.txt_source, "📄 Исходный текст")
        
        # Вкладка 2: Чанки
        self.txt_chunks = QTextEdit()
        self.txt_chunks.setReadOnly(True)
        self.txt_chunks.setFont(QFont("Menlo", 9))
        self.tabs.addTab(self.txt_chunks, "🔹 Чанки")
        
        # Вкладка 3: Поиск по индексу
        self.txt_search = QTextEdit()
        self.txt_search.setReadOnly(True)
        self.txt_search.setFont(QFont("Menlo", 9))
        self.tabs.addTab(self.txt_search, "🔍 Поиск")
        
        # Вкладка 4: Сущности
        self.txt_entities = QTextEdit()
        self.txt_entities.setReadOnly(True)
        self.txt_entities.setFont(QFont("Menlo", 9))
        self.tabs.addTab(self.txt_entities, "🏷️ Сущности")
        
        # Вкладка 5: Лог
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Menlo", 9))
        self.tabs.addTab(self.txt_log, "📋 Лог")
        
        right_layout.addWidget(self.tabs)
        
        # Поле поиска
        search_layout = QHBoxLayout()
        self.search_input = QTextEdit()
        self.search_input.setReadOnly(False)
        self.search_input.setPlaceholderText("Введите запрос для поиска...")
        self.search_input.setMaximumHeight(60)
        self.search_input.setFont(QFont("Arial", 11))
        search_layout.addWidget(self.search_input)
        
        self.btn_search = QPushButton("🔍 Найти")
        self.btn_search.setFixedHeight(60)
        self.btn_search.setFixedWidth(100)
        self.btn_search.clicked.connect(self._search)
        self.btn_search.setEnabled(False)
        search_layout.addWidget(self.btn_search)
        
        right_layout.addLayout(search_layout)

        # Кнопка очистки
        self.btn_clear = QPushButton("🗑️ Очистить")
        self.btn_clear.setFixedHeight(40)
        self.btn_clear.setStyleSheet("background-color: #c73e54;")
        self.btn_clear.clicked.connect(self._clear_all)
        right_layout.addWidget(self.btn_clear)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
    
    def _setup_styles(self):
        """Настройка стилей (CSS)"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            
            /* Заголовок */
            QLabel#title {
                color: #eee;
                font-size: 28px;
                font-weight: bold;
                padding: 10px 15px;
                background-color: #16213e;
                border-radius: 8px;
            }
            
            /* Кнопки */
            QPushButton {
                background-color: #0f3460;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e94560;
            }
            QPushButton:pressed {
                background-color: #c73e54;
            }
            QPushButton:disabled {
                background-color: #2a2a4a;
                color: #666;
            }
            
            /* Список файлов */
            QListWidget {
                background-color: #16213e;
                color: #eee;
                border: 2px solid #0f3460;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #0f3460;
            }
            QListWidget::item:hover {
                background-color: #1a3a5c;
            }
            
            /* Текстовые поля */
            QTextEdit {
                background-color: #16213e;
                color: #eee;
                border: 2px solid #0f3460;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 2px solid #e94560;
            }
            
            /* Прогресс бар */
            QProgressBar {
                background-color: #16213e;
                border: none;
                border-radius: 8px;
                text-align: center;
                color: #eee;
                font-weight: bold;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #e94560;
                border-radius: 8px;
            }
            
            /* Вкладки */
            QTabWidget::pane {
                border: 2px solid #0f3460;
                border-radius: 8px;
                background-color: #16213e;
            }
            QTabBar::tab {
                background-color: #1a1a2e;
                color: #888;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #16213e;
                color: #e94560;
            }
            QTabBar::tab:hover {
                color: #eee;
            }
            
            /* Счётчик */
            QLabel#file_count {
                color: #888;
                font-size: 12px;
                padding: 5px;
            }
            
            /* Статус бар */
            QStatusBar {
                background-color: #16213e;
                color: #888;
                border-top: 1px solid #0f3460;
            }
        """)
    
    def _load_files(self):
        """Открыть диалог выбора файлов"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите файлы",
            "",
            "Все файлы (*);;DOCX (*.docx);;PDF (*.pdf);;Изображения (*.png *.jpg *.jpeg)"
        )
        
        if files:
            self.file_paths.extend(files)
            self._update_file_list()
            self._log(f"Добавлено файлов: {len(files)}")
    
    def _update_file_list(self):
        """Обновить список файлов в UI"""
        self.file_list.clear()
        
        for file_path in self.file_paths:
            item = QListWidgetItem(Path(file_path).name)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            
            # Статус обработки
            if file_path in self.extracted_texts:
                item.setText(f"✅ {Path(file_path).name}")
            
            self.file_list.addItem(item)
        
        self.lbl_file_count.setText(f"Загружено файлов: {len(self.file_paths)}")
        self.btn_extract.setEnabled(len(self.file_paths) > 0)
    
    def _extract_text(self):
        """Извлечь текст из выбранных файлов"""
        if not self.file_paths:
            QMessageBox.warning(self, "Внимание", "Сначала выберите файлы!")
            return
        
        self._log(f"Начало извлечения текста из {len(self.file_paths)} файлов...")
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.file_paths))
        self.progress.setValue(0)
        self.btn_extract.setEnabled(False)
        self.statusBar().showMessage("Извлечение текста...")
        
        # Запускаем в отдельном потоке
        def progress_callback(current, total, filename):
            self.progress.setValue(current)
            self.statusBar().showMessage(f"Обработка: {Path(filename).name}")
        
        def on_finished(results):
            self.extracted_texts = results
            self._update_file_list()
            self.btn_extract.setEnabled(True)
            self.btn_process.setEnabled(True)
            self.progress.setVisible(False)
            self.statusBar().showMessage("Текст извлечён!")
            
            # Считаем общую статистику
            total_chars = sum(len(t) for t in results.values())
            total_chars_no_spaces = sum(len(t.replace(' ', '').replace('\n', '')) for t in results.values())
            total_words = sum(len(t.split()) for t in results.values())
            
            # Показываем первый файл
            if results:
                first_file = list(results.keys())[0]
                self._show_source_text(first_file, results[first_file])
                self._log(f"Извлечено текста: {total_chars} символов")
                self._log(f"Без пробелов: {total_chars_no_spaces} символов")
                self._log(f"Количество слов: {total_words}")
        
        def on_error(error):
            self._log(f"Ошибка: {error}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка извлечения текста:\n{error}")
            self.btn_extract.setEnabled(True)
            self.progress.setVisible(False)
        
        self.worker = WorkerThread(
            self.loader.load_multiple,
            self.file_paths,
            progress_callback=progress_callback
        )
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.worker.start()
    
    def _show_source_text(self, file_path: str, text: str):
        """Показать исходный текст"""
        # Показываем ВЕСЬ текст (без ограничений)
        
        html_text = f'''
        <div style="background-color: #1a1a2e; padding: 15px; border-radius: 8px;">
            <div style="color: #e94560; font-size: 16px; font-weight: bold; margin-bottom: 10px;">
                📄 {Path(file_path).name}
            </div>
            <div style="color: #eee; line-height: 1.6; white-space: pre-wrap; font-size: 11px;">
                {text}
            </div>
        </div>
        '''
        self.txt_source.setHtml(html_text)
    
    def _process_documents(self):
        """Обработать документы (очистка + чанки)"""
        if not self.extracted_texts:
            QMessageBox.warning(self, "Внимание", "Сначала извлеките текст!")
            return
        
        self._log("Начало обработки документов...")
        self.statusBar().showMessage("Обработка документов...")
        
        # Обрабатываем
        self.all_chunks = self.processor.process_multiple(self.extracted_texts)
        
        self._log(f"Создано чанков: {len(self.all_chunks)}")
        self.statusBar().showMessage(f"Готово! Чанков: {len(self.all_chunks)}")
        self.index_built = False  # Сбрасываем индекс
        
        # Показываем чанки
        self._show_chunks()
        
        # Включаем кнопку построения индекса
        self.btn_index.setEnabled(True)
        self.btn_entities.setEnabled(True)  # Включаем кнопку сущностей
    
    def _build_index(self):
        """Построить семантический индекс"""
        if not self.all_chunks:
            QMessageBox.warning(self, "Внимание", "Сначала обработайте документы!")
            return
        
        self._log("Построение семантического индекса...")
        self.statusBar().showMessage("Построение индекса...")
        self.btn_index.setEnabled(False)
        
        # Запускаем в отдельном потоке
        def on_finished(indexer):
            self.indexer = indexer
            self.index_built = True
            self.btn_index.setEnabled(False)  # Отключаем кнопку индекса (уже построен)
            self.btn_search.setEnabled(True)  # Включаем кнопку поиска
            self.statusBar().showMessage("Индекс построен!")
            
            stats = indexer.get_stats()
            self._log(f"Индекс построен: {stats['total_chunks']} чанков из {stats['total_files']} файлов")
            
            # Переключаемся на вкладку поиска
            self.tabs.setCurrentWidget(self.txt_search)
            self.txt_search.setText("Индекс готов! Введите запрос для поиска...")
        
        def on_error(error):
            self._log(f"Ошибка построения индекса: {error}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения индекса:\n{error}")
            self.btn_index.setEnabled(True)
        
        self.worker = WorkerThread(self._create_index)
        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.worker.start()
    
    def _create_index(self):
        """Создать индекс (вызывается в отдельном потоке)"""
        indexer = DocumentIndexer()
        indexer.build_index(self.all_chunks)
        return indexer
    
    def _extract_entities(self):
        """Извлечь сущности из чанков"""
        if not self.all_chunks:
            QMessageBox.warning(self, "Внимание", "Сначала обработайте документы!")
            return
        
        self._log("Извлечение сущностей...")
        self.statusBar().showMessage("Извлечение сущностей...")
        
        # Извлекаем сущности
        self.all_entities = self.entity_extractor.extract_from_chunks(self.all_chunks)
        
        stats = self.entity_extractor.get_stats(self.all_entities)
        self._log(f"Извлечено сущностей: {len(self.all_entities)}")
        for type_name, count in stats.items():
            self._log(f"  {type_name}: {count}")
        
        # Показываем сущности
        self._show_entities()
        
        self.statusBar().showMessage(f"Готово! Найдено {len(self.all_entities)} сущностей")
    
    def _show_entities(self):
        """Показать сущности в UI"""
        if not self.all_entities:
            self.txt_entities.setText("Сущности не найдены")
            return
        
        # Группируем по типам
        grouped = self.entity_extractor.group_by_type(self.all_entities)
        stats = self.entity_extractor.get_stats(self.all_entities)
        
        # Статистика
        stats_html = f'''
        <div style="background-color: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="color: #e94560; font-size: 18px; font-weight: bold;">🏷️ Статистика сущностей</div>
            <div style="color: #eee; margin-top: 10px;">
                Всего сущностей: <span style="color: #e94560; font-weight: bold;">{len(self.all_entities)}</span><br>
        '''
        
        for type_name, count in stats.items():
            stats_html += f'  {type_name}: <span style="color: #e94560; font-weight: bold;">{count}</span><br>'
        
        stats_html += '</div></div>'
        
        # Сущности по типам
        entities_html = ""
        type_colors = {
            "ФИО": "#e94560",
            "Технология": "#533483",
            "Термин": "#0f3460",
            "Дата": "#16a085",
            "Email": "#e67e22",
            "Телефон": "#27ae60",
            "URL": "#2980b9",
            "Организация": "#8e44ad",
        }
        
        for entity_type, entities in grouped.items():
            type_name = entity_type.value
            color = type_colors.get(type_name, "#888888")
            
            entities_html += f'''
            <div style="background-color: #1a1a2e; border-left: 4px solid {color}; padding: 12px; margin: 12px 0; border-radius: 4px;">
                <div style="color: {color}; font-size: 16px; font-weight: bold; margin-bottom: 10px;">
                    {type_name} ({len(entities)})
                </div>
            '''
            
            # Показываем первые 20 сущностей каждого типа
            for i, entity in enumerate(entities[:20]):
                entities_html += f'''
                <div style="background-color: #16213e; padding: 8px; margin: 6px 0; border-radius: 4px;">
                    <div style="color: #eee; font-size: 13px;">{entity.text}</div>
                    <div style="color: #666; font-size: 10px;">
                        {entity.source_file.split('/')[-1]} | Уверенность: {entity.confidence:.2f}
                    </div>
                </div>
                '''
            
            if len(entities) > 20:
                entities_html += f'<div style="color: #666; text-align: center; padding: 10px;">... и ещё {len(entities) - 20}</div>'
            
            entities_html += '</div>'
        
        self.txt_entities.setHtml(stats_html + entities_html)
    
    def _search(self):
        """Поиск по индексу"""
        query = self.search_input.toPlainText().strip()
        
        if not query:
            QMessageBox.warning(self, "Внимание", "Введите запрос!")
            return
        
        if not self.index_built:
            QMessageBox.warning(self, "Внимание", "Сначала постройте индекс!")
            return
        
        self._log(f"Поиск: {query}")
        
        try:
            results = self.indexer.search(query, top_k=10)
            
            if not results:
                self.txt_search.setText("Ничего не найдено.")
                return
            
            # Формируем вывод
            output = f'<div style="color: #888; margin-bottom: 15px;">Найдено {len(results)} результатов по запросу: "<span style="color: #e94560;">{query}</span>"</div>'
            
            for i, (chunk, score) in enumerate(results):
                relevance = "высокая" if score > 0.7 else "средняя" if score > 0.5 else "низкая"
                color = "#e94560" if score > 0.7 else "#533483" if score > 0.5 else "#0f3460"
                
                output += f'''
                <div style="background-color: #1a1a2e; border-left: 4px solid {color}; padding: 12px; margin: 12px 0; border-radius: 4px;">
                    <div style="color: {color}; font-size: 13px; font-weight: bold; margin-bottom: 8px;">
                        Результат {i+1} | Релевантность: {relevance} ({score:.3f})
                    </div>
                    <div style="color: #888; font-size: 11px; margin-bottom: 8px;">
                        Файл: {Path(chunk.source_file).name} | Позиция: {chunk.start_pos}-{chunk.end_pos}
                    </div>
                    <div style="color: #eee; line-height: 1.6; white-space: pre-wrap; font-size: 12px;">
                        {chunk.text}
                    </div>
                </div>
                '''
            
            self.txt_search.setHtml(output)
            self._log(f"Найдено {len(results)} результатов")
            
        except Exception as e:
            self._log(f"Ошибка поиска: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка поиска:\n{e}")
    
    def _show_chunks(self):
        """Показать чанки в UI"""
        if not self.all_chunks:
            self.txt_chunks.setText("Нет чанков")
            return
        
        # Статистика
        stats_html = f'''
        <div style="background-color: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="color: #e94560; font-size: 18px; font-weight: bold;">📊 Статистика чанков</div>
            <div style="color: #eee; margin-top: 10px;">
                Всего чанков: <span style="color: #e94560; font-weight: bold;">{len(self.all_chunks)}</span><br>
                Средний размер: <span style="color: #e94560; font-weight: bold;">{sum(len(c.text) for c in self.all_chunks) // len(self.all_chunks)}</span> символов
            </div>
        </div>
        '''
        
        chunks_html = ""
        colors = ["#e94560", "#0f3460", "#533483", "#16213e"]
        
        for i, chunk in enumerate(self.all_chunks):  # Показываем ВСЕ чанки
            color = colors[i % len(colors)]
            chunks_html += f'''
            <div style="background-color: #1a1a2e; border-left: 4px solid {color}; padding: 12px; margin: 12px 0; border-radius: 4px;">
                <div style="color: {color}; font-size: 14px; font-weight: bold; margin-bottom: 8px;">
                    Чанк {i+1} | Файл: {Path(chunk.source_file).name}
                </div>
                <div style="color: #888; font-size: 11px; margin-bottom: 10px;">
                    Позиция: {chunk.start_pos} - {chunk.end_pos} | Размер: {len(chunk.text)} символов
                </div>
                <div style="color: #eee; line-height: 1.6; white-space: pre-wrap; font-size: 12px;">
                    {chunk.text}
                </div>
            </div>
            '''
        
        self.txt_chunks.setHtml(stats_html + chunks_html)
    
    def _clear_all(self):
        """Очистить все данные"""
        self.file_paths = []
        self.extracted_texts = {}
        self.all_chunks = []
        self.all_entities = []
        self.index_built = False
        self.indexer = DocumentIndexer()
        self.entity_extractor = EntityExtractor(use_ner=False)
        
        self.file_list.clear()
        self.txt_source.clear()
        self.txt_chunks.clear()
        self.txt_search.clear()
        self.txt_entities.clear()
        self.lbl_file_count.setText("Загружено файлов: 0")
        self.btn_extract.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_index.setEnabled(False)
        self.btn_entities.setEnabled(False)
        self.btn_search.setEnabled(False)
        self.progress.setVisible(False)
        self.progress.setValue(0)
        
        self._log("Все данные очищены")
        self.statusBar().showMessage("Готов к работе")
    
    def _log(self, message: str):
        """Добавить сообщение в лог"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f'<span style="color: #e94560;">[{timestamp}]</span> <span style="color: #eee;">{message}</span>')
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )


def main():
    """Точка входа приложения"""
    app = QApplication(sys.argv)
    
    # Настройка шрифта приложения
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
