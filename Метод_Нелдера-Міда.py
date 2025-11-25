import sympy as sp # Імпортування бібліотеки sympy, необхідної у даній програмі для роботи з вводом рівнянь та підвищення інтерактивності

def get_xl_xg_xh(function_values):
    # Сортування індексів за значенням функції
    sorted_indices = sorted(range(len(function_values)), key=lambda i: function_values[i])
    xl = sorted_indices[0] # Індекс найменшої за значенням функції точки
    xg = sorted_indices[-2] # Індекс наступної за найбільшим значенням функції точки
    xh = sorted_indices[-1] # Індекс найбільшої за значенням функції точки
    return xl, xg, xh # Повернення результату
"""
Функція get_xl_xg_xh(function_values) виконує сортування індексів елементів за значенням.
У аргументах функція приймає список, утворений зі значеннями функції у точках симплексу.
Функція повертає індекси найменшого, найбільшого та наступного за значенням елементів.
"""

def calculating_delta(function_result, simplex):
    f_part = 0
    for result in function_result:
        f_part+=result/(len(simplex))
    delta = 0
    for result in function_result:
        delta += ((result-f_part)**2)/(len(simplex))
    delta_final = delta**0.5
    return delta_final
"""
Функція calculating_sigma(function_result, simplex), необхідна для обчислення критерію зупинки delta.
У аргументах функція приймає список, утворений зі значеннями функції у точках симплексу та самі точки симплексу.
Функція повертає обчислене значення delta.
"""

def calculating_func(function, simplex, var_names):
    result = []
    for point in simplex:
        subs_dict = {var_names[i]: point[i] for i in range(len(var_names))}
        result.append(function.subs(subs_dict))
    return result
"""
Функція calculating_func(function, simplex, var_names), необхідна для обчислення значення точок, що знаходяться у симплексі.
У аргументах функція приймає функцію, вписану користувачем, точки симплексу та імена змінних.
Функція повертає список значень функції при відповідних точках симплексу.
"""


def process(function_result, function, simplex, var_names, alpha, beta, gamma, epsilon, n):
    iter = 0 # Лічильник кількості ітерацій
    # Створення "нескінченного" циклу
    while True:
        iter = iter + 1
        if iter > 1000:
            print("-" * 60)
            print("[УВАГА] - Кількість ітерацій перевищила дозволену!\n(Схоже Ви натрапили на авражну функцію)")
            print("-" * 60)
            break
        print("-" * 60)
        print(f"ІТЕРАЦІЯ №{iter}")
        print("-" * 60)

        # Присвоєння змінним індексів найбільшої за значенням функції точки, наступної за найбільшим значенням та найменшим значенням точок
        xl_index, xg_index, xh_index = get_xl_xg_xh(function_result)
        x_l = simplex[xl_index]
        x_g = simplex[xg_index]
        x_h = simplex[xh_index]

        # Обчислення центру тяжіння всіх точок
        print("\nПроцес знаходження центру тяжіння всіх точок...")
        x_0 = [0] * len(simplex[0]) # Ініціалізація списку координат точки
        for i, point in enumerate(simplex):
            if i != xh_index:  # Виключення найбільшої за значенням функції точки
                for j in range(len(point)):
                    x_0[j] += point[j]
        x_0 = [x / (len(simplex) - 1) for x in x_0] # Створення координат точки
        print("\nx_0 = ", x_0)
        print("\nПроцес знаходження значення функції у центрі тяжіння всіх точок...")
        f_x_0 = calculating_func(function, [x_0], var_names)[0]
        print("\nf(x_0) = ", f_x_0)
        # Відбиття
        print("-" * 60)
        print("Процес відбиття...")
        print("-" * 60)
        x_r = [((1 + alpha) * x_0[i]) - (alpha * x_h[i]) for i in range(len(x_0))]
        f_r = calculating_func(function, [x_r], var_names)[0]
        f_l = calculating_func(function, [x_l], var_names)[0]
        f_g = calculating_func(function, [x_g], var_names)[0]
        f_h = calculating_func(function, [x_h], var_names)[0]
        print("\nПоточний симплекс:")
        for i, point in enumerate(simplex):
            print(f"x{i} = {point}")
        if f_r < f_l:
            # Розтягування
            print("-" * 60)
            print("Процес розтягування...")
            print("-" * 60)
            x_e = [x_0[i] + gamma * (x_r[i] - x_0[i]) for i in range(len(x_0))]
            f_e = calculating_func(function, [x_e], var_names)[0]

            if f_e < f_l:
                simplex[xh_index] = x_e
                function_result[xh_index] = f_e
                print("\nПоточний симплекс:")
                for i, point in enumerate(simplex):
                    print(f"x{i} = {point}")
            else:
                simplex[xh_index] = x_r
                function_result[xh_index] = f_r
                print("\nПоточний симплекс:")
                for i, point in enumerate(simplex):
                    print(f"x{i} = {point}")
            print("\nРозрахунок критерія зупинки delta...")
            delta = calculating_delta(function_result, simplex)
            if delta < epsilon:
                print("-" * 60)
                print("[УВАГА] - Вдалося досягти збіжності!")
                print("-" * 60)
                print("Мінімальна точка:", simplex[xl_index])
                print("delta =", delta)
                print("-" * 60)
                break
            else:
                print("-" * 60)
                print("[УВАГА] - Збіжності не досягнуто!\n(Перезапуск циклу...)")
                print("-" * 60)
                continue

        elif f_r > f_l and f_r <= f_g:
            simplex[xh_index] = x_r
            function_result[xh_index] = f_r
            print("\nПоточний симплекс:")
            for i, point in enumerate(simplex):
                print(f"x{i} = {point}")
            print("\nРозрахунок критерія зупинки delta...")
            delta = calculating_delta(function_result, simplex)
            if delta < epsilon:
                print("-" * 60)
                print("[УВАГА] - Вдалося досягти збіжності!")
                print("-" * 60)
                print("Мінімальна точка:", simplex[xl_index])
                print("delta =", delta)
                print("-" * 60)
                break
            else:
                print("-" * 60)
                print("[УВАГА] - Збіжності не досягнуто!\n(Перезапуск циклу...)")
                input("[УВАГА] - Натисніть клавішу Enter для продовження...")
                print("-" * 60)
                continue

        elif f_r > f_l and f_r > f_g:
            if f_r > f_h:
                # Стискання
                print("-" * 60)
                print("Процес стискання...")
                print("-" * 60)
                x_c = [(beta * x_h[i]) + ((1 - beta) * x_0[i]) for i in range(len(x_0))]
                f_c = calculating_func(function, [x_c], var_names)[0]

                if f_c < f_h:
                    simplex[xh_index] = x_c
                    function_result[xh_index] = f_c
                    print("\nПоточний симплекс:")
                    for i, point in enumerate(simplex):
                        print(f"x{i} = {point}")
                    print("\nРозрахунок критерія зупинки delta...")
                    delta = calculating_delta(function_result, simplex)
                    if delta < epsilon:
                        print("-" * 60)
                        print("[УВАГА] - Вдалося досягти збіжності!")
                        print("-" * 60)
                        print("Мінімальна точка:", simplex[xl_index])
                        print("delta =", delta)
                        print("-" * 60)
                        break
                    else:
                        print("-" * 60)
                        print("[УВАГА] - Збіжності не досягнуто!\n(Перезапуск циклу...)")
                        input("[УВАГА] - Натисніть клавішу Enter для продовження...")
                        print("-" * 60)
                        continue

                elif f_c > f_h:
                    for i in range(len(simplex)):
                        if i != xl_index:
                            for j in range(n):
                                simplex[i][j] = (simplex[i][j] + simplex[xl_index][j]) / 2
                    function_result = calculating_func(function, simplex, var_names)
                    print("\nПоточний симплекс:")
                    for i, point in enumerate(simplex):
                        print(f"x{i} = {point}")
                    print("\nРозрахунок критерія зупинки delta...")
                    delta = calculating_delta(function_result, simplex)
                    if delta < epsilon:
                        print("-" * 60)
                        print("[УВАГА] - Вдалося досягти збіжності!")
                        print("-" * 60)
                        print("Мінімальна точка:", simplex[xl_index])
                        print("delta =", delta)
                        print("-" * 60)
                        break
                    else:
                        print("-" * 60)
                        print("[УВАГА] - Збіжності не досягнуто!\n(Перезапуск циклу...)")
                        input("[УВАГА] - Натисніть клавішу Enter для продовження...")
                        print("-" * 60)
                        continue

            elif f_r < f_h:
                simplex[xh_index] = x_r
                function_result[xh_index] = f_r
                print("\nПоточний симплекс:")
                for i, point in enumerate(simplex):
                    print(f"x{i} = {point}")
                # Стискання
                print("-" * 60)
                print("Процес стискання...")
                print("-" * 60)
                x_c = [(beta * x_r[i]) + ((1 - beta) * x_0[i]) for i in range(len(x_0))]
                f_c = calculating_func(function, [x_c], var_names)[0]

                if f_c < f_h:
                    simplex[xh_index] = x_c
                    function_result[xh_index] = f_c
                    print("\nПоточний симплекс:")
                    for i, point in enumerate(simplex):
                        print(f"x{i} = {point}")
                    print("\nРозрахунок критерія зупинки delta...")
                    delta = calculating_delta(function_result, simplex)
                    if delta < epsilon:
                        print("-" * 60)
                        print("[УВАГА] - Вдалося досягти збіжності!")
                        print("-" * 60)
                        print("Мінімальна точка:", simplex[xl_index])
                        print("delta =", delta)
                        print("-" * 60)
                        break
                    else:
                        print("-" * 60)
                        print("[УВАГА] - Збіжності не досягнуто!\n(Перезапуск циклу...)")
                        input("[УВАГА] - Натисніть клавішу Enter для продовження...")
                        print("-" * 60)
                        continue

                elif f_c > f_h:
                    for i in range(len(simplex)):
                        if i != xl_index:
                            for j in range(n):
                                simplex[i][j] = (simplex[i][j] + simplex[xl_index][j]) / 2
                    print("\nПоточний симплекс:")
                    for i, point in enumerate(simplex):
                        print(f"x{i} = {point}")
                    function_result = calculating_func(function, simplex, var_names)
                    print("\nРозрахунок критерія зупинки delta...")
                    delta = calculating_delta(function_result, simplex)
                    if delta < epsilon:
                        print("-" * 60)
                        print("[УВАГА] - Вдалося досягти збіжності!")
                        print("-" * 60)
                        print("Мінімальна точка:", simplex[xl_index])
                        print("delta =", delta)
                        print("-" * 60)
                        break
                    else:
                        print("-" * 60)
                        print("[УВАГА] - Збіжності не досягнуто!\n(Перезапуск циклу...)")
                        input("[УВАГА] - Натисніть клавішу Enter для продовження...")
                        print("-" * 60)
                        continue
"""
Функція process(function_result, function, simplex, var_names, alpha, beta, gamma, epsilon, n) виконує основну частину алгоритму пошуку мінімуму нелінійної функції методом Нелдера-Міда.
У аргументах функція значення функції при початкових точках симплексу, сам симплекс, а також параметри введені користувачем.
Функція нічого не повертає, вона виводить на екран відразу результат.
"""

def main():
    print("~" * 15, "ЗНАХОДЖЕННЯ МІНІМУМУ НЕЛІНІЙНОЇ ФУНКЦІЇ МЕТОДОМ НЕЛДЕРА-МІДА", "~" * 15, "\n")
    print("-" * 60)
    n = int(input("Введіть від скількох змінних залежить нелінійне рівняння: "))
    print("-" * 60)
    var_names = input(f"Введіть позначення {n} змінних через кому: ").replace(" ", "").split(",")
    print("-" * 60)
    # Перевірка чи правильно користувач ввів позначення в програмі
    if len(var_names) != n:
        print("-" * 60)
        print("[ПОМИЛКА] - Кількість введених позначень змінних не відповідає числу n!")
        print("-" * 60)
        exit()
    # Створення математичних символів SymPy
    variables = sp.symbols(" ".join(var_names))
    # Пов'язання текст з математичними символами
    var_dict = {name: var for name, var in zip(var_names, variables)}
    print("-" * 60)
    print("Доступні позначення змінних:", ", ".join(var_names))
    print("-" * 60)
    user_input = input(f"Введіть рівняння f({', '.join(var_names)}): ")
    print("-" * 60)

    try:
        f = sp.sympify(user_input, locals=var_dict)
        # Перевірка правильності вводу рівняння
        used_symbols = {str(s) for s in f.free_symbols}
        if not used_symbols.issubset(set(var_names)):
            raise ValueError("[ПОМИЛКА] - Рівняння містить недозволені змінні!")

        print("[УВАГА] - Рівняння введено правильно!")
        print("-" * 60)
        # Введення параметрів програми
        alpha = float(input("Введіть коефіцієнт відбиття alpha: "))
        print("-" * 60)
        while alpha < 0:
            alpha = float(input("[УВАГА] - Введіть ще раз коефіцієнт відбиття alpha: "))
            print("-" * 60)
        beta = float(input("Введіть коефіцієнт стискання beta: "))
        print("-" * 60)
        while beta < 0:
            beta = float(input("[УВАГА] - Введіть ще раз коефіцієнт стискання beta: "))
            print("-" * 60)
        gamma = float(input("Введіть коефіцієнт розтягування gamma: "))
        print("-" * 60)
        while gamma < 0:
            gamma = float(input("[УВАГА] - Введіть ще раз коефіцієнт розтягування gamma: "))
            print("-" * 60)
        h = float(input("Введіть початкове значення кроку h: "))
        print("-" * 60)
        while h < 0:
            h = float(input("[УВАГА] - Введіть ще раз початкове значення кроку: "))
            print("-" * 60)
        epsilon = float(input("Введіть точність обчислень epsilon: "))
        print("-" * 60)
        while epsilon < 0:
            epsilon = float(input("[УВАГА] - Введіть ще раз точність обчислень epsilon: "))
            print("-" * 60)
        print("Введіть початкову точку:")
        # Створення множини початкових точок
        start_point = {}

        for name in var_names:
            value = float(input(f"{name}0 = "))
            start_point[name] = value
        print("-" * 60)
        print("\nПочаткова точка:", start_point)
        print(f"\n{"~" * 60}\nДАНО:\nФункція {n} змінних: {f}\nalpha: {alpha}\nbeta: {beta}\ngamma: {gamma}\nh: {h}\nepsilon: {epsilon}\n{"~" * 60}")

        print("\nОбчислення прирощень sigma1 та sigma2...")
    except Exception as e:
        print("\n[Помилка] - Деталі:", e)
        exit()
    # Обчислення прирощень
    sigma1 = ((((n + 1) ** 0.5) + n - 1) / (n * (2 ** 0.5))) * h
    sigma2 = ((((n + 1) ** 0.5) - 1) / (n * (2 ** 0.5))) * h
    print(f"sigma_1 = {sigma1}")
    print(f"sigma_2 = {sigma2}")

    # Стартовий симплекс: список точок
    print("\nФормування початкового симплексу...")
    simplex = [] # Створення списку для збереження точок симплекса

    # Додання першої, стартової точки до симплекса
    x0 = [start_point[name] for name in var_names]
    simplex.append(x0)

    # Формування наступних точок
    for i in range(n):
        new_point = x0.copy()
        for j in range(n):
            if i == j:
                new_point[j] += sigma1
            else:
                new_point[j] += sigma2
        simplex.append(new_point)
    print("\nСимплекс сформовано:")
    for i, point in enumerate(simplex):
        print(f"x{i} = {point}")
    print("\nОбчислення значення в точках симплексу...")
    function_result = calculating_func(f, simplex, var_names)
    for i, point in enumerate(function_result):
        print(f"f({', '.join(var_names)}){i} = {point}")
    print("\nЗапуск процесу знаходження мінімуму нелінійного рівняння...")
    process(function_result, f, simplex, var_names, alpha, beta, gamma, epsilon, n)
"""
Функція main() - серце програми.
У ній виконується введення параметрів користувачем, а також їх перевірка, побудова початкового симплексу і значень функції у ньому, 
а також виконує виклик функції process.
"""

main() # Виклик функції main()

