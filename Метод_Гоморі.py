import numpy as np # Підключення бібліотеки numpy для полегшення роботи з симплекс-таблицею
import math # Імпортування стандартного математичного модуля

# Ініціалізація дуже малих чисел для відсікання зайвого шуму
TOL = 1e-9
PIV_TOL = 1e-12

def frac_part(val):
    """
    Функція frac_part обчислює дробову частину дійсного числа і гарантує, що результат належить інтервалу [0; 1),
    навіть якщо число дуже близьке до цілого і містить похибки обчислень з плаваючою комою.
    Функція приймає число, для якого обчислюється дробова частина.
    Повертається коректна дробова частина у діапазоні [0; 1).
    """

    f = val - math.floor(val + 1e-12) # Виділення дробової частини
    # Якщо є похибка, то необхідно заглушити шум
    if abs(f - 1.0) < 1e-10:
        return 0.0
    if abs(f) < 1e-12:
        return 0.0

    return f # Повернення результату

def remove_error(tab, eps=1e-12):
    """
    Функція clean_small прибирає мікрошум після операцій з опорним елементом у симплекс-таблиці (числова стабільність).
    Функція приймає таблицю та дуже мале значення для визначення похибки.
    Функція нічого не повертає, а змінює на місці симплекс-таблицю.
    """

    tab[np.abs(tab) < eps] = 0.0

def pivot_operation(tab, r, c):
    """
    Функція pivot_operation виконує операції над опорним елементом в симплекс-таблиці.
    Функція приймає поточну симплекс-таблицю, опорні рядок і стовпчик.
    Функція нічого не повертає, а змінює на місці симплекс-таблицю.
    """

    piv = tab[r, c] # Знайдення опорного елемента
    tab[r, :] /= piv # Ділення опорного рядка на значення опорного елемента
    for rr in range(tab.shape[0]): # Цикл по всіх рядках
        if rr == r: # Якщо знайдено опорний рядок, то треба його пропустити
            continue
        mult = tab[rr, c] # Виділення елемента, що знаходиться нижче чи вище опорного елемента в тому ж стовпці
        if abs(mult) > PIV_TOL: # Якщо елемент ненульовий
            tab[rr, :] -= mult * tab[r, :] # Застосування правила трикутника
    remove_error(tab) # Прибирання зайвого накопиченого шуму

def fmt_num(x):
    """
    Функція fmt_num потрібна для того, щоб красиво надрукувати число в симплекс таблиці.
    Функція приймає число з цієї таблиці.
    Повертає відформатований рядок.
    """

    if abs(x) < 1e-10: # Якщо у числі наявний мікрошум, то просто треба його онулити
        x = 0.0
    return f"{x: .6f}" # Повернення результату

def show_table(tab, bas, var_labels, caption=""):
    """
    Функція show_table друкує таблицю на кожному кроці перетворення симплекс-таблиці.
    Функція приймає аргументами саму симплекс-таблицю, масив, що зберігає індекси лише тих змінних, що входять до базису
    та масив, що зберігає символьні позначення усіх змінних, за можливості додається ще заголовок.
    """

    if caption: # Надрукувати заголовок таблиці, якщо він існує
        print("\n", f"{caption}:", "\n")

    # Підписи шапки таблиці
    cols = ["БАЗИС"] + var_labels + ["ВІЛЬН. ЧЛЕН"]

    # Обчислення ширини колонок
    widths = []

    # Формування ширини стовпчика "БАЗИС"
    basis_vals = [var_labels[bas[i]] for i in range(tab.shape[0] - 1)] + ["F"]
    widths.append(max(len("БАЗИС"), max(len(s) for s in basis_vals), 6) + 2) # Формування ширини стовпчика

    # Формування ширини стовпчиків змінних
    for j in range(tab.shape[1] - 1):
        header_w = len(var_labels[j])
        data_w = max(len(fmt_num(tab[i, j])) for i in range(tab.shape[0]))
        widths.append(max(header_w, data_w, 6) + 2) # Формування ширини стовпчиків

    # Формування ширини стовпчика "ВІЛЬНИЙ ЧЛЕН"
    rhs_w = max(len("ВІЛЬН. ЧЛЕН"), max(len(fmt_num(tab[i, -1])) for i in range(tab.shape[0])))
    widths.append(rhs_w + 2) # Формування ширини стовпчика

    total_w = sum(widths) + len(widths) + 1 # Формування загальної ширини

    # Виведення шапки
    print("|" + "|".join(h.center(w) for h, w in zip(cols, widths)) + "|")
    print("|" + "-" * (total_w - 2) + "|")

    # Виведення рядків обмежень
    for i in range(tab.shape[0] - 1):
        row = [var_labels[bas[i]]] + [fmt_num(tab[i, j]) for j in range(tab.shape[1] - 1)] + [fmt_num(tab[i, -1])]
        print("|" + "|".join(v.center(w) for v, w in zip(row, widths)) + "|")

    # Виведення рядка цільової функції
    print("|" + "-" * (total_w - 2) + "|")
    obj_row = ["F"] + [fmt_num(tab[-1, j]) for j in range(tab.shape[1] - 1)] + [fmt_num(tab[-1, -1])]
    print("|" + "|".join(v.center(w) for v, w in zip(obj_row, widths)) + "|")


def primal_simplex(tab, bas, labels, log=True, it_cap=200):
    """
    Функція primal_simplex виконує прямий симплекс.
    Аргументами функції є: поточна симплекс-таблиця, bas — список індексів базисних змінних, labels — назви змінних для виводу (x1, x2, …),
    log — чи друкувати проміжні таблиці та повідомлення, it_cap — максимальна кількість ітерацій (щоб уникнути зациклення).
    Функція повертає значення булевого типу True або False, а також змінює симплексну таблицю.
    """

    step = 0 # Кількість ітерацій
    while step < it_cap: # Основний цикл
        step += 1 # Збільшення кроку

        obj_row = tab[-1, :-1] # Останній рядок таблиці
        enter_col = int(np.argmin(obj_row))  # Найвід'ємніше значення рядка F

        if obj_row[enter_col] >= -1e-10: # Якщо найвід’ємніше значення вже не від’ємне, то поточний план є оптимальним
            if log:
                print("\n[УВАГА] - Оптимум знайдено прямим симплексом!")
            return True

        col = tab[:-1, enter_col] # Cтовпець вхідної змінної тільки у рядках обмежень
        rhs = tab[:-1, -1] # Cтовпець вільних членів у рядках обмежень

        ratio_list = []
        # Застосування правила мінімального відношення
        for i, a in enumerate(col):
            if a > TOL:
                ratio_list.append((rhs[i] / a, i))
        # Якщо нема жодного додатного a у стовпці вхідної змінної
        if not ratio_list:
            print("\n[ПОМИЛКА] - Задача необмежена (прямий симплекс)!")
            return False

        _, leave_row = min(ratio_list) # Вибір вихідного рядка

        if log:
            print(f"\n|-> Прямий симплекс | Крок {step}:\n|-> Вхідна змінна = {labels[enter_col]}\n|-> Вихідна змінна = {labels[bas[leave_row]]}")
        # Перетворення таблиці
        pivot_operation(tab, leave_row, enter_col)
        bas[leave_row] = enter_col

        if log:
            show_table(tab, bas, labels, caption=f"Таблиця після опорного кроку (крок {step})")

    print("\n[ПОМИЛКА] - Перевищено кількість ітерацій у прямому симплексі!")
    return False

def dual_simplex(tab, bas, labels, log=True, it_cap=300):
    """
    Функція dual_simplex виконує двоїстий симплекс.
    Функція приймає ті самі аргументи, що й функція primal_simplex
    """

    step = 0 # Кількість ітерацій
    while step < it_cap: # Основний цикл
        step += 1 # Збільшення кроку

        rhs = tab[:-1, -1] # Вибір стовпця вільних членів тільки для обмежень
        leave_row = int(np.argmin(rhs)) # Знаходження найбільш від'ємного значення

        if rhs[leave_row] >= -1e-10: # Якщо навіть мінімальний вільний член вже не від’ємний, то розв’язок допустимий
            if log:
                print("\n[УВАГА] - Допустимість відновлено (двоїстий симплекс)!")
            return True

        # Пошук кандидатів на вхідний стовпець
        row = tab[leave_row, :-1]
        candidates = []
        for j, a in enumerate(row):
            if a < -TOL:
                # Мінімізація
                candidates.append((tab[-1, j] / (-a), j))

        if not candidates:
            print("\n[ПОМИЛКА] - Двоїстий симплекс: немає дозволеного вхідного стовпця!")
            return False

        _, enter_col = min(candidates) # Вибір вхідного стовпця

        if log:
            print(f"\n|-> Двоїстий симплекс | Крок {step}:\n|-> Вихідна змінна = {labels[bas[leave_row]]}\n|-> Вхідна змінна = {labels[enter_col]}")

        # Перетворення таблиці
        pivot_operation(tab, leave_row, enter_col)
        bas[leave_row] = enter_col

        if log:
            show_table(tab, bas, labels, caption=f"Таблиця після опорного кроку двоїстого методу (крок {step})")

    print("\n[ПОМИЛКА] - Перевищено кількість ітерацій у двоїстому симплексі!")
    return False

def extract_solution(tab, bas, total_vars):
    """
    Функція extract_solution витягує поточний розв’язок із симплекс-таблиці після виконання симплекс-методу.
    Функція приймає tab — поточна симплекс-таблиця, bas — список індексів базисних змінних,
    total_vars — загальна кількість змінних у задачі.
    Функція повертає вектор значень змінних та значення цільової функції.
    """

    sol = np.zeros(total_vars) # Створення масиву довжини total_vars, заповненого нулями
    # Заповнення базисних змінних
    for i, bi in enumerate(bas):
        sol[bi] = tab[i, -1]
    obj_val = tab[-1, -1] # Отримання значення цільової функції

    return sol, obj_val # Повернення результату

def gomory_cut(tab, bas, labels, integer_var_count, log=True):
    """
    Додає відсікання Гоморі.
    Бере рядок з базисною змінною серед integer-змінних, де RHS має найбільшу дробову частину.
    """

    m = tab.shape[0] - 1 # Кількість базисних змінних
    n = tab.shape[1] - 1 # Кількість змінних

    # Кандидати: рядки, де базисна змінна належить до "цілих" і вільний член дробовий
    cand = []
    for i in range(m):
        bi = bas[i]
        if bi < integer_var_count:
            f = frac_part(tab[i, -1])
            cand.append((f, i))

    if not cand:
        return None
    # Вибір найкращого рядка
    best_frac, src_row = max(cand, key=lambda t: t[0])
    if best_frac <= 1e-10:
        return None  # Вже цілі

    # Зберігання рядка до модифікації таблиці
    row_before = tab[src_row, :].copy()

    # Побудова коефіцієнтів відсікання Гоморі
    cut_a = np.array([-frac_part(row_before[j]) for j in range(n)], dtype=float)
    cut_b = -frac_part(row_before[-1])

    g_count = sum(1 for nm in labels if nm.startswith("k"))
    new_g = f"k{g_count + 1}"

    labels.insert(n, new_g)

    # Додавання нового стовпець у таблицю
    extra_col = np.zeros((tab.shape[0], 1))
    tab = np.hstack([tab[:, :n], extra_col, tab[:, n:]])

    # Формування нового рядка обмежень
    new_row = np.zeros((1, tab.shape[1]))
    new_row[0, :n] = cut_a
    new_row[0, n] = 1.0
    new_row[0, -1] = cut_b

    tab = np.vstack([tab[:-1, :], new_row, tab[-1:, :]])
    bas.append(n)

    if log:
        print("\n|-> Відсікання Гоморі")
        print(f"|-> Обрано рядок базису: {labels[bas[src_row]]} (дробова частина вільного члена = {best_frac:.6f})")
        # Показ у зручній формі: sum frac(a_j) * x_j >= frac(b)
        terms = []
        for j in range(n):
            fj = frac_part(row_before[j])
            if fj > 1e-10:
                terms.append(f"{fj:.6f} * {labels[j]}")
        print("|-> Відсікання (у вигляді): " + " + ".join(terms) + f" >= {frac_part(row_before[-1]):.6f}")
        show_table(tab, bas, labels, caption="Таблиця після додавання відсікання")

    return tab, bas, labels

def gomory(show=True, cut_cap=30):
    """
    Функція gomory запускає весь алгоритм методу відсікання Гоморі.
    Функція приймає аргументами show — якщо True, програма друкує заголовки, таблиці та коментарі,
    cut_cap — максимальна кількість відсікання (щоб не було нескінченних спроб).
    """
    labels = ["x1", "x2", "x3", "x4", "x5"] # Список назв змінних

    # Матриця коефіцієнтів обмежень
    A = np.array([
        [ 6,  4, 1, 0, 0],
        [ 3, -3, 0, 1, 0],
        [-1,  3, 0, 0, 1],
    ], dtype=float)

    b = np.array([24, 9, 3], dtype=float) # Вектор вільних членів

    c = np.array([2, 1, 0, 0, 0], dtype=float) # Коефіцієнти цільової функції

    integer_var_count = 5 # Кількість цілих змінних

    # Створення симплекс-таблиці
    m, n = A.shape
    tab = np.zeros((m + 1, n + 1), dtype=float)
    tab[:m, :n] = A
    tab[:m, -1] = b
    tab[-1, :n] = -c

    bas = [2, 3, 4] # Початковий базис

    if show:
        print("\n\t|", "~" * 57, "|")
        print("\t| РОЗВ'ЯЗАННЯ ЦІЛОЧИСЕЛЬНОЇ ЗАДАЧІ МЕТОДОМ ВІДСІКАНЬ ГОМОРІ |")
        print("\t|", "~" * 57, "|")
        show_table(tab, bas, labels, caption="Початкова симплекс-таблиця")

    # Прямий симплекс
    if not primal_simplex(tab, bas, labels, log=show):
        return

    for k in range(1, cut_cap + 1): # Основний цикл додавання відсікання
        sol, obj = extract_solution(tab, bas, len(labels)) # Витяг розв’язку

        # Перевірка цілочисельності
        integer_ok = True
        for j in range(integer_var_count):
            if abs(sol[j] - round(sol[j])) > 1e-7:
                integer_ok = False
                break

        if integer_ok:
            if show:
                print("\n[УВАГА] - Цілочисельний оптимум знайдено!")
            sol_int = np.round(sol[:integer_var_count]).astype(int)
            x1, x2, x3, x4, x5 = sol_int.tolist()
            print("\n\t\t\t\t|", "~" * 9, "|")
            print("\t\t\t\t| РЕЗУЛЬТАТ |")
            print("\t\t\t\t|", "~" * 9, "|")
            print("\n|-> Відповідь:")
            print(f"x1 = {x1}, x2 = {x2}, x3 = {x3}, x4 = {x4}, x5 = {x5}")
            print(f"F = 2 * x1 + x2 = {2*x1 + x2}")
            return

        if show:
            print("\n" + "~" * 45)
            print(f"|-> Відсікання №{k} (бо розв’язок ще не цілий)")
            print("~" * 45)
        # Якщо розв’язок не цілий - додання відсікання
        upd = gomory_cut(tab, bas, labels, integer_var_count=integer_var_count, log=show)
        if upd is None:
            print("\n[ПОМИЛКА] - Не знайшлось рядка для коректного відсікання (вже майже ціле або нестандартний випадок)!")
            break

        tab, bas, labels = upd

        if not dual_simplex(tab, bas, labels, log=show):
            break

    print("\n[ПОМИЛКА] - Не вдалося знайти цілочисельний оптимум!")

def main():
    """
    Головна функція програми.
    """
    gomory(show=True, cut_cap=30) # Виклик функції реалізації алгоритму методу Гоморі

main() # Виклик функції main
