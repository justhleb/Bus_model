# Bus System Simulation

## Описание

Эта программа моделирует автобусную систему с несколькими остановками и рейсами, анализируя загруженность автобусов и время ожидания пассажиров. Она генерирует визуализации в формате PDF, включая гистограммы, boxplot-диаграммы и текстовые метрики для каждой остановки, а также обобщённые данные по времени ожидания и загруженности автобусов.

### В модели присутствует ряд упрощений. 

**1)** Мы считаем, что количество людей, направляющихся на все остановки, распределено равномерно.

**2)** Пассажиры всегда дожидаются автобуса и не уходят, даже если ждут долго.

**3)** У автобуса фиксированное количество мест, и (потесниться) не получится.

### Основные возможности
- **Симуляция**: Моделирует прибытие пассажиров на остановки и движение автобусов с учётом вместимости.
- **Визуализация**:
  - Титульная страница с основными параметрами модели.
  - Гистограммы, boxplot и метрики (среднее, медиана, максимум/минимум с временем) для:
    - Количества пассажиров на каждой остановке (10 страниц).
    - Среднего времени ожидания пассажиров (1 страница).
    - Средней загруженности автобусов (1 страница).
- **Формат вывода**: PDF-файл

## Требования

- **Python**: 3.6 или выше.
- **Библиотеки**:
  - `pandas` — для работы с данными CSV.
  - `numpy` — для вычислений.
  - `matplotlib` — для создания графиков и PDF.
- **Входные файлы**:
  - `passengers.csv`: Данные о количестве пассажиров на остановках по времени.
  - `buses.csv`: Расписание автобусов (время прибытия на каждую остановку).

## Установка

1. Убедитесь, что Python установлен:
   ```bash
   python --version
   ```
2. Установите необходимые библиотеки:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. Поместите файлы `passengers.csv` и `buses.csv` в ту же директорию, где находится скрипт.

## Использование

1. Сохраните код в файл, например, `bus_simulation.py`.
2. Убедитесь, что входные файлы `passengers.csv` и `buses.csv` доступны.
3. Запустите программу:
   ```bash
   python bus_simulation.py
   ```
4. После выполнения будут созданы:
   - `passenger_histograms.pdf` — файл с визуализациями.
   - Вывод в консоль: средняя загруженность автобусов и среднее время ожидания.

### Параметры программы
- `passenger_file = "passengers.csv"`: Имя файла с данными о пассажирах.
- `bus_file = "buses.csv"`: Имя файла с расписанием автобусов.
- `bus_capacity = 50`: Вместимость автобуса (можно изменить в коде).
- `pdf_output = "passenger_histograms.pdf"`: Имя выходного PDF-файла.

## Структура входных файлов

### `passengers.csv`
- **Формат**: CSV с количеством пассажиров по остановкам и времени.
- **Столбцы**: Временные интервалы (например, "05:00", "05:10", ...).
- **Строки**: Остановки (`stop1`, `stop2`, ..., `stop10`).
- **Значения**: Целые числа (например, 2–10 пассажиров).

Пример:
```
,05:00,05:10,05:20,...
stop1,2,3,4,...
stop2,3,2,5,...
...
```

### `buses.csv`
- **Формат**: CSV с расписанием автобусов.
- **Столбцы**: Остановки (`stop1`, `stop2`, ..., `stop10`).
- **Строки**: Автобусы (`bus1`, `bus2`, ..., `bus40`).
- **Значения**: Время прибытия в формате "HH:MM".

Пример:
```
,stop1,stop2,stop3,...
bus1,05:00,05:06,05:12,...
bus2,05:26,05:32,05:38,...
...
```

## Вывод программы

### Консоль
- Средняя загруженность автобусов (например, "14.50 пассажиров").
- Среднее время ожидания (например, "4.50 минут").

Пример:
```
Средняя загруженность автобусов: 14.50 пассажиров
Среднее время ожидания: 4.50 минут
```

### `passenger_histograms.pdf`
- **Страница 0**: Титульная страница с параметрами модели:
  ```
  Основные параметры модели
  Вместимость автобуса: 50 пассажиров
  Количество рейсов: 40
  Времена начала рейсов: 05:00 - 22:00 (интервал: 26 мин 9 сек)
  Количество остановок: 10
  ```
- **Страницы 1–10**: Для каждой остановки (`stop1`–`stop10`):
  - Гистограмма (голубая): Количество пассажиров по времени.
  - Boxplot (голубой): Распределение пассажиров.
  - Текст: Среднее, медиана, максимум/минимум с временем.
  Пример:
  ```
  Среднее: 4.85
  Медиана: 4.00
  Максимум: 10 в 08:00
  Минимум: 2 в 05:00
  ```
- **Страница 11**: Среднее время ожидания:
  - Гистограмма (зелёная), boxplot (зелёный), текст.
  Пример:
  ```
  Среднее: 4.50 мин
  Медиана: 4.20 мин
  Максимум: 7.80 мин в 08:00
  Минимум: 2.10 мин в 14:00
  ```
- **Страница 12**: Средняя загруженность автобусов:
  - Гистограмма (оранжевая), boxplot (оранжевый), текст.
  Пример:
  ```
  Среднее: 14.50 пасс.
  Медиана: 13.80 пасс.
  Максимум: 22.30 пасс. в 08:10
  Минимум: 5.40 пасс. в 05:00
  ```

## Примечания
- **Зависимости**: Убедитесь, что все библиотеки установлены.
- **Входные данные**: Если `passengers.csv` или `buses.csv` отсутствуют, программа завершится с ошибкой. Эти файлы должны быть предварительно созданы.
- **Настройка**: Измените `bus_capacity` в `__main__` для симуляции с другой вместимостью.

## Ответ на задание 2
В данной модели используется дискретно-событийный подход, так как все изменения системы происходят в дискретные моменты времени, которые связаны с конкретными событиями (прибытие автобуса на остановку). В течение поездки пассажиры не могут выходить или входить в автобус, а значит, нам нет смысла рассматривать процесс как непрерывный и достаточно просчитывать только изменения на остановках. Если усложнять модель, к примеру, добавить пассажирам возможность уходить с остановки в случае слишком долгого ожидания или добавить автобусам адаптивное поведение на случай различных ситуаций на дорогах (ДТП, прорыв трубы на маршруте и т.д.), то можно будет сделать гибрид из дискретно-событийного и агентного подхода. Но в данной программе представлен вариант именно дискретно-событийного подхода.
