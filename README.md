# GeoRAG AI Backend 🧠

Мощный бэкенд на базе FastAPI для индексации документов и ответов на вопросы с использованием гибридного RAG.

## 📋 Требования

- Python 3.10+
- PostgreSQL 14+ с расширением `pgvector`
- Библиотека `poppler-utils` (для обработки PDF)

## 🛠 Установка

1. **Создайте окружение и установите зависимости:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Настройка переменных окружения:**
   Создайте файл `.env` в корне этой папки:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/geoscan
   GROQ_API_KEY=your_key_here
   EMBEDDING_DIMENSIONS=1024
   DATABASE_ENABLED=true
   ```

3. **Системные зависимости (Linux):**
   ```bash
   sudo apt-get install poppler-utils
   ```

## 🚀 Запуск

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Бэкенд автоматически инициализирует базу данных и создаст необходимые расширения при первом запуске (если у пользователя БД есть права суперпользователя).

## 📄 API Endpoints

- `GET /api/v1/documents`: Список всех документов.
- `POST /api/v1/documents`: Загрузка нового файла.
- `DELETE /api/v1/documents/{id}`: Удаление файла.
- `POST /api/v1/chat`: Задать вопрос (RAG).
- `GET /api/v1/graph`: Получить структуру графа знаний.
- `GET /api/v1/scenarios`: Демо-сценарии для быстрой проверки.
