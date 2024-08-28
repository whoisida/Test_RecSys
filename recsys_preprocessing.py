
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def precision_at_k(actual, predicted, k):
    # Убедимся, что predicted — это плоский список
    predicted_flat = [item for sublist in predicted for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    actual_set = set(actual)
    predicted_set = set(predicted_flat[:k])
    return len(actual_set & predicted_set) / float(k)


# Функция для вычисления Recall@k
def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))

# Функция для вычисления Average Precision (AP) @k
def average_precision_at_k(actual, predicted, k=10):
    if not actual:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

# Функция для оценки метрик top 10

def recommend_categories_top(item_name, df_dum, top_n=10):
    # Проверяем, что категория есть в данных
    if item_name not in df_dum.columns:
        raise ValueError(f"Категория '{item_name}' не найдена в данных.")
    
    # Находим все чеки, где была указана категория
    checks_with_item = df_dum[df_dum[item_name] > 0]
    
    # Считаем, сколько раз встречаются остальные категории
    category_sums = checks_with_item.sum(axis=0)
    
    # Убираем саму категорию из результатов
    category_sums = category_sums.drop(item_name)
    
    # Сортируем по убыванию частоты встречаемости
    recommended_categories = category_sums.sort_values(ascending=False).head(top_n)
    
    # Возвращаем список рекомендованных категорий
    return recommended_categories.index.tolist()

# Функция для оценки метрик



def evaluate_recommendations_top_10(test_df, df_dum, top_k=10):
    precisions = []
    recalls = []
    maps = []

    # Функция для очистки названий категорий
    def clean_category_name(category_name):
        return category_name.replace(' ', '_').replace(',', '_').replace('.', '_') \
                            .replace('мл', '').replace(r'\d+', '', regex=True).strip()

    for check_id in test_df.index:
        # Получаем фактические категории товаров из чека
        actual_items = test_df.columns[test_df.loc[check_id].values > 0].tolist()
        
        if not actual_items:
            continue

        # Попробуем несколько категорий для генерации рекомендаций
        for _ in range(len(actual_items)):
            # Возьмем любую категорию из чека для генерации рекомендаций
            random_item = np.random.choice(actual_items)
            # Очищаем название категории
            random_item_clean = clean_category_name(random_item)
            
            # Проверяем, что очищенное название категории существует в данных
            if random_item_clean in df_dum.columns:
                # Получаем рекомендации для очищенного названия
                predicted_items = recommend_categories_top(random_item_clean, df_dum, top_n=top_k)
                
                # Рассчитываем метрики
                precisions.append(precision_at_k(actual_items, predicted_items, top_k))
                recalls.append(recall_at_k(actual_items, predicted_items, top_k))
                maps.append(average_precision_at_k(actual_items, predicted_items, top_k))
                break
            else:
                print(f"Категория '{random_item_clean}' не найдена в данных. Пробуем другую категорию.")
        else:
            print(f"Все категории для чека {check_id} отсутствуют в данных.")

    # Выводим средние значения метрик
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_map = np.mean(maps)
    
    print(f'Средняя Precision@{top_k}: {mean_precision}')
    print(f'Средняя Recall@{top_k}: {mean_recall}')
    print(f'Средняя MAP@{top_k}: {mean_map}')
    
    return mean_precision, mean_recall, mean_map





######################
def recommend_categories_cos(item_name, similarity_matrix, top_n=10):
    similar_items = similarity_matrix[item_name].sort_values(ascending=False)
    return similar_items.index[1:top_n+1]

def evaluate_recommendations_cos(test_df, train_sim_df, top_k=10):
    precisions = []
    recalls = []
    maps = []

    for check_id in test_df.index:
        # Получаем фактические категории товаров из чека
        actual_items = test_df.columns[test_df.loc[check_id].values > 0].tolist()
        
        if not actual_items:
            continue

        # Возьмем любую категорию из чека для генерации рекомендаций
        random_item = np.random.choice(actual_items)
        predicted_items = recommend_categories_cos(random_item, train_sim_df).tolist()

        # Рассчитываем метрики
        precisions.append(precision_at_k(actual_items, predicted_items, top_k))
        recalls.append(recall_at_k(actual_items, predicted_items, top_k))
        maps.append(average_precision_at_k(actual_items, predicted_items, top_k))

    # Выводим средние значения метрик
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_map = np.mean(maps)

    print(f'Средняя Precision@{top_k}: {mean_precision}')
    print(f'Средняя Recall@{top_k}: {mean_recall}')
    print(f'Средняя MAP@{top_k}: {mean_map}')

    return mean_precision, mean_recall, mean_map






################################


def recommend_categories_svd(item_name, similarity_matrix, top_n=10):
    # Проверяем, что категория есть в данных
    if item_name not in similarity_matrix.columns:
        raise ValueError(f"Категория '{item_name}' не найдена в данных.")
    
    # Извлекаем схожие категории для item_name
    similar_items = similarity_matrix[item_name].sort_values(ascending=False)
    
    # Убираем саму категорию из результатов
    similar_items = similar_items.drop(item_name)
    
    # Возвращаем топ-N рекомендованных категорий как список строк
    return similar_items.head(top_n).index.tolist()






def evaluate_recommendations_svd(test_df, category_sim_df, top_k=10):
    precisions = []
    recalls = []
    maps = []

    # Проходим по каждому чеку в тестовом наборе
    for check_id in test_df.index:
        # Получаем фактические категории товаров из чека
        actual_items = test_df.columns[test_df.loc[check_id].values > 0].tolist()
        
        if not actual_items:
            continue
        
        # Выбираем случайную категорию из чека для генерации рекомендаций
        random_item = np.random.choice(actual_items)
        
        # Генерируем рекомендации для этой категории
        predicted_items = recommend_categories_svd(random_item, category_sim_df, top_n=top_k)
        
        # Рассчитываем метрики
        precisions.append(precision_at_k(actual_items, predicted_items, top_k))
        recalls.append(recall_at_k(actual_items, predicted_items, top_k))
        maps.append(average_precision_at_k(actual_items, predicted_items, top_k))

    # Возвращаем средние значения метрик
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_map = np.mean(maps)

    print(f'Средняя Precision@{top_k}: {mean_precision}')
    print(f'Средняя Recall@{top_k}: {mean_recall}')
    print(f'Средняя MAP@{top_k}: {mean_map}')

    return mean_precision, mean_recall, mean_map







