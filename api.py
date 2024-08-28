


from fastapi import FastAPI, HTTPException, Query
import pickle
import pandas as pd
from fastapi.responses import JSONResponse
import urllib.parse

# Функция для декодирования строк в UTF-8, если они закодированы неправильно
def decode_to_utf8(text):
    if isinstance(text, bytes):
        return text.decode('utf-8')
    return text

# Функция для очистки ввода пользователя (замена подчеркиваний на пробелы)
def clean_category_name(category_name):
    return category_name.replace("_", " ").replace(",", "_").replace(".", "_")

# Загружаем сохраненную матрицу схожести категорий
with open("category_similarity_matrix.pkl", "rb") as f:
    category_similarity_df = pickle.load(f)

# Применяем декодирование для всех строк (если это необходимо)
category_similarity_df.columns = [decode_to_utf8(col) for col in category_similarity_df.columns]
category_similarity_df.index = [decode_to_utf8(idx) for idx in category_similarity_df.index]

# Создаем приложение FastAPI
app = FastAPI()

# Функция для получения рекомендаций
def recommend_categories(item_name, similarity_matrix, top_n=10):
    # Проверяем, что категория есть в данных
    if item_name not in similarity_matrix.columns:
        raise ValueError(f"Категория '{item_name}' не найдена в данных.")
    
    # Извлекаем схожие категории для item_name
    similar_items = similarity_matrix[item_name].sort_values(ascending=False)
    
    # Убираем саму категорию из результатов
    similar_items = similar_items.drop(item_name)
    
    # Возвращаем топ-N рекомендованных категорий
    return similar_items.head(top_n).index.tolist()

# Создаем эндпоинт для получения рекомендаций
@app.get("/recommendations/")
def get_recommendations(category: str = Query(..., description="Название категории", example="КИСЛОМОЛОЧНЫЕ_ПРОДУКТЫ"), top_n: int = 10):
    try:
        # Декодируем категорию из URL-encoding
        decoded_category = urllib.parse.unquote(category)
        
        # Очищаем и нормализуем название категории (замена подчеркиваний на пробелы)
        cleaned_category = clean_category_name(decoded_category)
        
        # Возвращаем рекомендации для введенной категории
        recommendations = recommend_categories(cleaned_category, category_similarity_df, top_n)

        # Преобразуем выходные данные в корректный формат с кодировкой UTF-8
        decoded_recommendations = [decode_to_utf8(rec) for rec in recommendations]
        
        # Возвращаем JSON с правильной кодировкой UTF-8
        return JSONResponse(content={"input_category": cleaned_category, "recommendations": decoded_recommendations}, media_type="application/json; charset=utf-8")
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
