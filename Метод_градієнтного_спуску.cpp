// Метод_градієнтного_спуску.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// Підключення бібліотек
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include <conio.h>

float function(float x, float y); // Прототип функції, що зберігає в собі цільову функцію

void grad(float x, float y, float* gx, float* gy); // Прототип функції, необхідної для обчислення градієнта функції f

float fi(float lambda, float x, float y, float dx, float dy); // Прототип функції, необхідної для обчислення виразу градієнта функції 

float delta(float a, float fa, float b, float fb, float c, float fc); // Прототип функції, необхідної для обчислення вершини параболи

float interpolation(float x, float y, float dx, float dy, float lambda0); // Прототип функції, необхідної для реалізації методу градієнтного спуску

void gradient_descent(float x0, float y0, float eps, int steps, float lambda0); // Прототип функції, що реалізовує метод градієнтного спуску

int main()
{
    SetConsoleOutputCP(1251); // Налаштування кодування консолі для коректного виводу кирилиці
    printf("\n\t|--------------------------------------------------------------------------------------------------------------------------|");
    printf("\n\t|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ЗНАХОДЖЕННЯ ЕКСТРЕМУМУ ФУНКЦІЇ МЕТОДОМ ГРАДІЄНТНОГО СПУСКУ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|\n");
    printf("\t|--------------------------------------------------------------------------------------------------------------------------|\n\n");
    float x0, y0, epsilon, lambda;
    printf("\n\t\t|---------------------------------------------------------------------------------------------------------|");
    printf("\n\t\t|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ УМОВА ТА ВСТАНОВЛЕННЯ ПАРАМЕТРІВ ПРОГРАМИ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|\n");
    printf("\t\t|---------------------------------------------------------------------------------------------------------|\n\n");

    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    printf("\nЗадана функція: f(x,y) = (x - 2)^2 + 2 * y^2 - 10\n");
    printf("\nВкажіть координати початкової точки (x0 y0):\n");
    printf("\nx0 = ");
    if (scanf_s("%f", &x0) != 1)
    {
        printf("\n[ПОМИЛКА] - Помилка вводу!\n");
        return 1;
    }
    printf("\ny0 = ");
    if (scanf_s("%f", &y0) != 1)
    {
        printf("\n[ПОМИЛКА] - Помилка вводу!\n");
        return 1;
    }
    printf("\nВведіть точність розрахунків epsilon: ");
    if (scanf_s("%f", &epsilon) != 1)
    {
        printf("\n[ПОМИЛКА] - Помилка вводу!\n");
        return 1;
    }
    while (epsilon <= 0)
    {
        printf("\n[УВАГА] - Введіть ще раз точність розрахунків epsilon: ");
        if (scanf_s("%f", &epsilon) != 1)
        {
            printf("\n[ПОМИЛКА] - Помилка вводу!\n");
            return 1;
        }
    }
    printf("\nВведіть коефіцієнт lambda: ");
    if (scanf_s("%f", &lambda) != 1)
    {
        printf("\n[ПОМИЛКА] - Помилка вводу!\n");
        return 1;
    }
    while (lambda <= 0)
    {
        printf("\nВведіть ще раз коефіцієнт lambda: ");
        if (scanf_s("%f", &lambda) != 1)
        {
            printf("\n[ПОМИЛКА] - Помилка вводу!\n");
            return 1;
        }
    }
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    gradient_descent(x0, y0, epsilon, 100, lambda);

    return 0; // Успішне завершення роботи програми
}

float function(float x, float y) // Визначення функції, що зберігає в собі цільову функцію
{
    return (x - 2.0f) * (x - 2.0f) + 2.0f * y * y - 10.0f;
}

void grad(float x, float y, float* gx, float* gy) // Визначення функції, необхідної для обчислення градієнта функції f
{
    *gx = 2.0f * (x - 2.0f); // Часткова похідна по x
    *gy = 4.0f * y; // Часткова похідна по y
}

float fi(float lambda, float x, float y, float dx, float dy) // Визначення функції, необхідної для обчислення виразу градієнта функції 
{
    return function(x + lambda * dx, y + lambda * dy);
}

float delta(float a, float fa, float b, float fb, float c, float fc) // Визначення функції, необхідної для обчислення вершини параболи
{
    float num, den;
    num = (b * b - c * c) * fa + (c * c - a * a) * fb + (a * a - b * b) * fc; // Чисельник
    den = (b - c) * fa + (c - a) * fb + (a - b) * fc; // Знаменник

    if (fabsf(den) < 1e-7f)
    {
        return b; // Повернення середньої точки, якщо знаменник дуже малий
    }
    return 0.5f * (num / den); // Повернення вершини параболи
}

float interpolation(float x, float y, float dx, float dy, float lambda0) // Визначення функції, необхідної для реалізації методу градієнтного спуску
{
    // Початкові точки
    float a = 0;
    float b = lambda0 / 2;
    float c = lambda0;

    // Значення функції у початкових точках
    float fa = fi(a, x, y, dx, dy);
    float fb = fi(b, x, y, dx, dy);
    float fc = fi(c, x, y, dx, dy);

    float tmp, d, fd, nt[3], nv[3];

    int worst, kk;

    for (int it = 1; it <= 3; it++) // 3 ітерації інтерполяції
    {
        d = delta(a, fa, b, fb, c, fc);
        fd = fi(d, x, y, dx, dy);


        float t[4] = { a, b, c, d };
        float v[4] = { fa, fb, fc, fd };
        worst = 0;
        // Знаходження найгіршої точки
        for (int i = 1; i < 4; i++)
        {
            if (v[i] > v[worst])
            {
                worst = i;
            }
        }

        kk = 0;
        // Створення масиву кращих точок
        for (int i = 0; i < 4; i++)
        {
            if (i == worst)
            {
                continue;
            }
            nt[kk] = t[i];
            nv[kk] = v[i];
            kk++;
        }

        // Сортування точок
        for (int i = 0; i < 3; i++)
        {
            for (int j = i + 1; j < 3; j++)
            {
                if (nt[j] < nt[i])
                {
                    tmp = nt[i];
                    nt[i] = nt[j];
                    nt[j] = tmp;

                    tmp = nv[i];
                    nv[i] = nv[j];
                    nv[j] = tmp;
                }
            }
        }

        a = nt[0];
        fa = nv[0];
        b = nt[1];
        fb = nv[1];
        c = nt[2];
        fc = nv[2];

        printf("   [LS] ітерація %d: a = %.4f b = %.4f c = %.4f | fi(b) = %.6f\n", it, a, b, c, fb);
    }

    return b; // Оптимальний крок
}

void gradient_descent(float x0, float y0, float eps, int steps, float lambda0) // Визначення функції, що реалізовує метод градієнтного спуску
{
    float x = x0, y = y0, lambda = lambda0, gx, gy, dx, dy, module_grad, lam;

	for (int k = 1; k <= steps; k++) // Основний цикл методу градієнтного спуску
    {
        printf("\n\t\t|----------------------------------------------------------------------------|");
        printf("\n\t\t|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ІТЕРАЦІЯ %d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|\n", k);
        printf("\t\t|----------------------------------------------------------------------------|\n\n");
        grad(x, y, &gx, &gy);

        dx = -gx;
        dy = -gy;

        module_grad = sqrtf(dx * dx + dy * dy);

        printf("====================================================================\n");
        printf("Точка: x = %.6f y = %.6f\n\n", x, y);
        printf("f = %.6f\n\n", function(x, y));
        printf("grad=(%.6f, %.6f) | |d| = %.6f\n\n", gx, gy, module_grad);

		if (module_grad < eps) // Умова зупинки
        {
            printf("Зупинка: |d| < eps (%.6f < %.6f)\n\n", module_grad, eps);
            printf("====================================================================\n");
            break;
        }

        printf("Старе значення lambda = %.4f\n\n", lambda);

        lam = interpolation(x, y, dx, dy, lambda);
        printf("\nНове значення lambda* = %.6f\n\n", lam);

        x = x + lam * dx;
        y = y + lam * dy;

        printf("Нова точка: x = %.6f y = %.6f\n", x, y);
        printf("Нове f = %.6f\n", function(x, y));
        printf("====================================================================\n");
		lambda *= 0.5; // Зменшення кроку для наступної ітерації
        puts("\nНатисніть клавішу, щоб продовжити...\n");
        _getch();
    }
    printf("\n\t\t|-------------------------------------------------------------------------|");
    printf("\n\t\t|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ РЕЗУЛЬТАТ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|\n");
    printf("\t\t|-------------------------------------------------------------------------|\n\n");
    printf("\n-------------------------\n");
    printf("x* = (%.6f, %.6f)\n", x, y);
    printf("f_min = %.6f\n", function(x, y));
    printf("-------------------------\n");
}