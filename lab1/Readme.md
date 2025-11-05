## Эндпоинты
- `GET /health` — проверка сервиса
- `GET /model_info` — информация о модели
- `POST /predict` — предсказание для одного текста
  ```json
  { "text": "нормальный текст" }
- `POST /predict_batch` — предсказание для списка текстов
  ```json
  { "texts": ["текст 1", "текст 2"] }
    ```

## Локальный запуск
```uvicorn app.main:app --reload```

```http://127.0.0.1:8000/docs```


## Docker
```bash
docker build -t toxicity-api .
docker run --rm -p 8000:8000 toxicity-api
```
```
http://127.0.0.1:8000/docs
```
## Тесты
```bash
pytest -q
```


