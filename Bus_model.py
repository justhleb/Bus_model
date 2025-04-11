import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def parse_time(t):
    """Преобразует строку времени (HH:MM) в минуты с начала дня."""
    return datetime.strptime(t, "%H:%M").hour * 60 + datetime.strptime(t, "%H:%M").minute

def plot_passenger_histograms(passenger_file, pdf_filename, waiting_data, bus_loads, bus_capacity):
    """Создаёт титульную страницу и гистограммы с boxplot и метриками."""
    passengers_df = pd.read_csv(passenger_file, index_col=0)
    buses_df = pd.read_csv(bus_file, index_col=0)
    stops = passengers_df.index.tolist()
    time_cols = passengers_df.columns.tolist()
    time_minutes = [parse_time(t) for t in time_cols]
    
    with PdfPages(pdf_filename) as pdf:
        # Титульная страница
        fig = plt.figure(figsize=(14, 6))
        fig.text(0.5, 0.9, 'Основные параметры модели', ha='center', va='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.7, f'Вместимость автобуса: {bus_capacity} пассажиров', ha='center', va='center')
        fig.text(0.5, 0.6, f'Количество рейсов: {len(buses_df.index)}', ha='center', va='center')
        first_bus_time = buses_df.iloc[0, 0]
        last_bus_time = buses_df.iloc[-1, 0]
        interval_minutes = (parse_time(last_bus_time) - parse_time(first_bus_time)) / (len(buses_df.index) - 1)
        interval_str = f"{int(interval_minutes)} мин {int((interval_minutes % 1) * 60)} сек"
        fig.text(0.5, 0.5, f'Времена начала рейсов: {first_bus_time} - {last_bus_time} (интервал: {interval_str})', ha='center', va='center')
        fig.text(0.5, 0.4, f'Количество остановок: {len(stops)}', ha='center', va='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Гистограммы пассажиров для каждой остановки с boxplot и метриками
        for stop in stops:
            passengers = passengers_df.loc[stop].values
            avg_passengers = np.mean(passengers)
            median_passengers = np.median(passengers)
            max_passengers = np.max(passengers)
            max_time = time_cols[np.argmax(passengers)]
            min_passengers = np.min(passengers)
            min_time = time_cols[np.argmin(passengers)]
            
            fig = plt.figure(figsize=(14, 6))
            
            # Гистограмма
            ax1 = fig.add_axes([0.1, 0.15, 0.45, 0.75])
            ax1.bar(time_minutes, passengers, width=10, align='edge', color='skyblue', edgecolor='black')
            ax1.set_xlabel('Время (HH:MM)')
            ax1.set_ylabel('Количество пассажиров')
            ax1.set_title(f'Пассажиры на остановке {stop}')
            xticks = np.arange(parse_time("05:00"), parse_time("00:00") + 24 * 60, 60)
            xtick_labels = [f"{h // 60:02d}:{h % 60:02d}" for h in xticks]
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xtick_labels, rotation=45)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Boxplot 
            ax2 = fig.add_axes([0.6, 0.15, 0.15, 0.75])
            ax2.boxplot(passengers, vert=True, patch_artist=True, boxprops=dict(facecolor='skyblue'))
            ax2.set_xticks([])
            ax2.set_ylabel('Количество пассажиров')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Текстовые метрики
            text_x = 0.8
            text_y_start = 0.8
            text_y_step = 0.1
            fig.text(text_x, text_y_start, f'Среднее: {avg_passengers:.2f}', ha='left')
            fig.text(text_x, text_y_start - text_y_step, f'Медиана: {median_passengers:.2f}', ha='left')
            fig.text(text_x, text_y_start - 2 * text_y_step, f'Максимум: {max_passengers} в {max_time}', ha='left')
            fig.text(text_x, text_y_start - 3 * text_y_step, f'Минимум: {min_passengers} в {min_time}', ha='left')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Гистограмма среднего времени ожидания с boxplot и метриками
        if waiting_data:
            bins = np.arange(parse_time("05:00"), parse_time("00:00") + 24 * 60 + 10, 10)
            bin_indices = np.digitize([t for t, _ in waiting_data], bins, right=True)
            bin_means = []
            for i in range(1, len(bins)):
                times_in_bin = [w for j, (_, w) in enumerate(waiting_data) if bin_indices[j] == i]
                bin_means.append(np.mean(times_in_bin) if times_in_bin else 0)
            
            valid_means = [m for m in bin_means if m > 0]
            avg_waiting = np.mean(valid_means) if valid_means else 0
            median_waiting = np.median(valid_means) if valid_means else 0
            max_waiting = np.max(valid_means) if valid_means else 0
            max_waiting_idx = np.argmax(bin_means) if any(m > 0 for m in bin_means) else 0
            max_waiting_time = time_cols[max_waiting_idx] if max_waiting_idx < len(time_cols) else time_cols[0]
            min_waiting = np.min([m for m in valid_means if m > 0]) if valid_means else 0
            min_waiting_idx = np.argmin([m if m > 0 else float('inf') for m in bin_means])
            min_waiting_time = time_cols[min_waiting_idx] if min_waiting_idx < len(time_cols) else time_cols[0]
            
            fig = plt.figure(figsize=(14, 6))
            
            # Гистограмма 
            ax1 = fig.add_axes([0.1, 0.15, 0.45, 0.75])
            ax1.bar(bins[:-1], bin_means, width=10, align='edge', color='lightgreen', edgecolor='black')
            ax1.set_xlabel('Время прихода (HH:MM)')
            ax1.set_ylabel('Среднее время ожидания (минуты)')
            ax1.set_title('Среднее время ожидания по времени прихода')
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xtick_labels, rotation=45)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Boxplot 
            ax2 = fig.add_axes([0.6, 0.15, 0.15, 0.75])
            ax2.boxplot(valid_means, vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
            ax2.set_xticks([])
            ax2.set_ylabel('Среднее время ожидания (минуты)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Текстовые метрики 
            text_x = 0.8
            fig.text(text_x, text_y_start, f'Среднее: {avg_waiting:.2f} мин', ha='left')
            fig.text(text_x, text_y_start - text_y_step, f'Медиана: {median_waiting:.2f} мин', ha='left')
            fig.text(text_x, text_y_start - 2 * text_y_step, f'Максимум: {max_waiting:.2f} мин в {max_waiting_time}', ha='left')
            fig.text(text_x, text_y_start - 3 * text_y_step, f'Минимум: {min_waiting:.2f} мин в {min_waiting_time}', ha='left')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Гистограмма средней загруженности автобусов с boxplot и метриками
        if bus_loads:
            bins = np.arange(parse_time("05:00"), parse_time("00:00") + 24 * 60 + 10, 10)
            bin_sums = [0] * (len(bins) - 1)
            bin_counts = [0] * (len(bins) - 1)
            for _, time, load in bus_loads:
                bin_idx = np.digitize(time, bins, right=True) - 1
                if 0 <= bin_idx < len(bin_sums):
                    bin_sums[bin_idx] += load
                    bin_counts[bin_idx] += 1
            bin_means = [s / c if c > 0 else 0 for s, c in zip(bin_sums, bin_counts)]
            
            valid_means = [m for m in bin_means if m > 0]
            avg_load = np.mean(valid_means) if valid_means else 0
            median_load = np.median(valid_means) if valid_means else 0
            max_load = np.max(valid_means) if valid_means else 0
            max_load_idx = np.argmax(bin_means) if any(m > 0 for m in bin_means) else 0
            max_load_time = time_cols[max_load_idx] if max_load_idx < len(time_cols) else time_cols[0]
            min_load = np.min([m for m in valid_means if m > 0]) if valid_means else 0
            min_load_idx = np.argmin([m if m > 0 else float('inf') for m in bin_means])
            min_load_time = time_cols[min_load_idx] if min_load_idx < len(time_cols) else time_cols[0]
            
            fig = plt.figure(figsize=(14, 6))
            
            # Гистограмма 
            ax1 = fig.add_axes([0.1, 0.15, 0.45, 0.75])
            ax1.bar(bins[:-1], bin_means, width=10, align='edge', color='orange', edgecolor='black')
            ax1.set_xlabel('Время (HH:MM)')
            ax1.set_ylabel('Средняя загруженность (пассажиры)')
            ax1.set_title('Средняя загруженность автобусов по времени')
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xtick_labels, rotation=45)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Boxplot 
            ax2 = fig.add_axes([0.6, 0.15, 0.15, 0.75])
            ax2.boxplot(valid_means, vert=True, patch_artist=True, boxprops=dict(facecolor='orange'))
            ax2.set_xticks([])
            ax2.set_ylabel('Средняя загруженность (пассажиры)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Текстовые метрики 
            text_x = 0.8
            fig.text(text_x, text_y_start, f'Среднее: {avg_load:.2f} пасс.', ha='left')
            fig.text(text_x, text_y_start - text_y_step, f'Медиана: {median_load:.2f} пасс.', ha='left')
            fig.text(text_x, text_y_start - 2 * text_y_step, f'Максимум: {max_load:.2f} пасс. в {max_load_time}', ha='left')
            fig.text(text_x, text_y_start - 3 * text_y_step, f'Минимум: {min_load:.2f} пасс. в {min_load_time}', ha='left')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def simulate_bus_system(passenger_file, bus_file, bus_capacity):
    """Симулирует автобусную систему, учитывая прибытие пассажиров и движение автобусов."""
    # Чтение данных о пассажирах и автобусах из CSV-файлов
    passengers_df = pd.read_csv(passenger_file, index_col=0)  # Пассажиры по остановкам и времени
    buses_df = pd.read_csv(bus_file, index_col=0)  # Расписание автобусов
    time_cols = [parse_time(col) for col in passengers_df.columns]  # Временные метки в минутах
    start_time = parse_time("05:00")  # Начало дня
    end_time = parse_time("00:00") + 24 * 60  # Конец дня (следующие сутки)
    stops = passengers_df.index.tolist()  # Список остановок
    num_stops = len(stops)  # Количество остановок
    
    # Инициализация структуры для ожидающих пассажиров на каждой остановке
    waiting_passengers = {stop: [] for stop in stops}  # [время_прибытия, id_пассажира]
    passenger_id = 0  # Уникальный идентификатор пассажира
    
    # Генерация пассажиров на остановках
    for stop in stops:
        for time_idx, time in enumerate(time_cols):
            num_passengers = int(passengers_df.iloc[stops.index(stop), time_idx])  # Число пассажиров в интервале
            for _ in range(num_passengers):
                # Случайное время прибытия в пределах 10-минутного интервала
                arrival_time = time + random.uniform(0, 10)
                waiting_passengers[stop].append([arrival_time, passenger_id])
                passenger_id += 1
    
    # Сортировка пассажиров по времени прибытия на каждой остановке
    for stop in stops:
        waiting_passengers[stop].sort(key=lambda x: x[0])
    
    # Инициализация списков для записи данных симуляции
    bus_loads = []  # [bus_idx, arrival_time, current_passengers] - загруженность автобусов
    waiting_data = []  # [arrival_time, waiting_time] - время ожидания пассажиров
    
    # Симуляция движения каждого автобуса
    for bus_idx in buses_df.index:
        current_passengers = 0  # Текущее число пассажиров в автобусе
        bus_times = [parse_time(buses_df.loc[bus_idx, stop]) for stop in stops]  # Времена прибытия на остановки
        
        # Обработка каждой остановки для текущего автобуса
        for stop_idx, stop in enumerate(stops):
            arrival_time = bus_times[stop_idx]  # Время прибытия автобуса на остановку
            # Запись текущей загруженности перед высадкой/посадкой
            bus_loads.append([bus_idx, arrival_time, current_passengers])
            
            # Высадка пассажиров (случайное число, но не больше текущей загрузки)
            if stop_idx > 0:  # На первой остановке никто не выходит
                num_exiting = random.randint(0, min(current_passengers, bus_capacity))
                current_passengers -= num_exiting
            
            # Посадка пассажиров
            free_seats = bus_capacity - current_passengers  # Свободные места
            stop_passengers = waiting_passengers[stop]  # Список ожидающих на остановке
            boarding_passengers = []  # Пассажиры, которые сядут
            remaining_passengers = []  # Пассажиры, которые останутся ждать
            
            # Проверка каждого пассажира на остановке
            for passenger in stop_passengers:
                if passenger[0] <= arrival_time and free_seats > 0:  # Пассажир успел и есть место
                    waiting_time = arrival_time - passenger[0]  # Время ожидания
                    waiting_data.append([passenger[0], waiting_time])  # Запись данных
                    boarding_passengers.append(passenger)
                    free_seats -= 1
                    current_passengers += 1
                else:
                    remaining_passengers.append(passenger)  # Пассажир остаётся ждать
            
            # Обновление списка ожидающих на остановке
            waiting_passengers[stop] = remaining_passengers
    
    # Расчёт итоговых метрик
    avg_load = np.mean([load[2] for load in bus_loads]) if bus_loads else 0  # Средняя загруженность
    avg_waiting_time = np.mean([w for _, w in waiting_data]) if waiting_data else 0  # Среднее время ожидания
    return avg_load, avg_waiting_time, waiting_data, bus_loads

if __name__ == "__main__":
    passenger_file = "passengers.csv"
    bus_file = "buses.csv"
    bus_capacity = 50
    pdf_output = "passenger_histograms.pdf"
    
    # Запуск симуляции и визуализации
    avg_load, avg_waiting_time, waiting_data, bus_loads = simulate_bus_system(passenger_file, bus_file, bus_capacity)
    plot_passenger_histograms(passenger_file, pdf_output, waiting_data, bus_loads, bus_capacity)