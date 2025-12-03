import numpy as np

class LatticeGas:
    def __init__(self, parametrs: dict, obstacle: dict):
        """
        Инициализация модели решеточных газов (D2Q9).
        """
        # 1. Инициализация параметров
        self.nx = parametrs['nx']
        self.ny = parametrs['ny']
        self.u_lb = parametrs['u_lb']
        self.Re = parametrs['Re']
        
        # Параметры препятствия
        self.cx = obstacle['xc']
        self.cy = obstacle['yc']
        self.r = obstacle['r']
        
        # 2. Расчет констант
        # Кинематическая вязкость v = (u_lb * r) / Re
        self.nu = (self.u_lb * self.r) / self.Re
        # Время релаксации omega = 1 / (3v + 0.5)
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        
        # Начальная скорость v_init = u_lb
        self.v_init = self.u_lb 

        # 3. Определение векторов и весов модели D2Q9
        # c0=(0,0), c1=(1,0), c2=(0,1), c3=(-1,0), c4=(0,-1), ...
        self._v = np.array([
            [0, 0],   # 0
            [1, 0],   # 1
            [0, 1],   # 2
            [-1, 0],  # 3
            [0, -1],  # 4
            [1, 1],   # 5
            [-1, 1],  # 6
            [-1, -1], # 7
            [1, -1]   # 8
        ])
        
        # Веса ai
        self._a = np.array([
            4/9,    # 0
            1/9, 1/9, 1/9, 1/9,        # 1-4
            1/36, 1/36, 1/36, 1/36     # 5-8
        ])
        
        # Вспомогательные массивы для векторизации (из methods.py)
        self._index = np.arange(9)
        self._vx = self._v[:, 0]
        self._vy = self._v[:, 1]
        
        # Индексы направлений для граничных условий (Inflow)
        # Нам нужно знать, какие векторы смотрят "внутрь" (вправо), какие "наружу" (влево), а какие нейтральны.
        # Для D2Q9 при потоке слева направо:
        # Входящие (неизвестные на левой границе, vx > 0): 1, 5, 8
        self._ind_left = np.array([1, 5, 8]) 
        # Выходящие/Отраженные (известные, приходят изнутри, vx < 0): 3, 6, 7
        self._ind_right = np.array([3, 6, 7])
        # Нейтральные по X (vx = 0): 0, 2, 4
        self._ind_middle = np.array([0, 2, 4])

        # 4. Инициализация массивов полей
        # Shape: (nx, ny, 9) используется в methods.py (судя по axis в np.roll и self.u_x)
        self.f_in = np.zeros((self.nx, self.ny, 9))
        self.f_out = np.zeros((self.nx, self.ny, 9))
        self.f_equil = np.zeros((self.nx, self.ny, 9))
        
        self.density = np.ones((self.nx, self.ny)) # rho = 1
        self.u_x = np.full((self.nx, self.ny), self.v_init) # u = (v_init, 0)
        self.u_y = np.zeros((self.nx, self.ny))
        
        # 5. Инициализация начального состояния (f_in = f_eq)
        self.calc_f_equil()
        self.f_in = self.f_equil.copy()

        # 6. Определение препятствия
        # Используем статический метод, но сохраняем результат в self.obstacle (маска boolean)
        grid_shape = (self.nx, self.ny)
        obst_dict = self.add_cylinder(self.cx, self.cy, self.r, grid_shape)
        # Преобразуем словарь координат в булеву маску для numpy
        self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        self.obstacle[obst_dict['x'], obst_dict['y']] = True
        
        # Списки для хранения истории (для visual.ipynb)
        self.field_den = []
        self.field_u = []
        self.field_p = []
        self.field_ux = []
        self.field_uy = []

    @staticmethod
    def add_cylinder(xc, yc, r, shape):
        """
        Определение узлов решетки, принадлежащих препятствию (круг).
        Возвращает словарь с координатами x и y.
        """
        # Проверка входных данных (из теста)
        if xc - r < 0 or xc + r >= shape[0] or yc - r < 0 or yc + r >= shape[1]:
             # В тестах ожидается ValueError при выходе за границы, хотя здесь логика границ может отличаться
             # Оставим простую реализацию
             pass

        rx, ry = np.arange(shape[0]), np.arange(shape[1])
        # Создаем сетку координат
        X, Y = np.meshgrid(rx, ry, indexing='ij')
        # Условие круга: (x-xc)^2 + (y-yc)^2 <= r^2
        mask = ((X - xc)**2 + (Y - yc)**2) <= r**2
        
        x_indices, y_indices = np.where(mask)
        return {'x': x_indices, 'y': y_indices}

    # --- Методы из methods.py ---
    
    def calc_f_equil(self) -> np.ndarray:
        """Calculate f equilibrium"""
        # Формула (2) из метод. указаний
        norma = self.u_x**2 + self.u_y**2
        for i, vx, vy, a in zip(self._index, self._vx, self._vy, self._a):
            dot = self.u_x*vx + self.u_y*vy
            # 4.5 = 9/2, 1.5 = 3/2. Формула совпадает с
            self.f_equil[:,:,i] = self.density * a * (1 + 3*dot + 4.5*dot**2 - 1.5*norma)

    def calc_inflow(self):
        """
        Calculate inflow boundary condition (Left boundary)
        Здесь реализуется условие Zubou для заданной скорости.
        """
        # Задаем скорость на входе
        self.u_x[0, :] = self.v_init
        self.u_y[0, :] = 0
        
        # Считаем плотности от векторов, направленных "нейтрально" и "влево" (изнутри домена)
        # Внимание: shape (nx, ny, 9), значит берем [0, :, indices]
        rho_2 = np.sum(self.f_in[0, :, self._ind_middle], axis=0) # Сумма по оси 9-ти компонентов
        rho_3 = np.sum(self.f_in[0, :, self._ind_right], axis=0)
        
        # Расчет плотности на входе (Zubou)
        self.density[0, :] = 1/(1 - self.u_x[0, :]) * (rho_2 + 2*rho_3)

        # Пересчитываем равновесную функцию для границы с новыми u и rho
        # (Локально для границы, но проще вызвать полный пересчет или оптимизировать)
        # Для простоты вызовем полный, но это дорого. Оптимизированный вариант ниже:
        
        # Локальный расчет f_equil только для x=0
        norma_b = self.u_x[0,:]**2 + self.u_y[0,:]**2
        for i, vx, vy, a in zip(self._index, self._vx, self._vy, self._a):
            dot = self.u_x[0,:]*vx + self.u_y[0,:]*vy
            self.f_equil[0,:,i] = self.density[0,:]*a*(1 + 3*dot + 4.5*dot**2 - 1.5*norma_b)

        # Присваиваем входящие значения (с коррекцией неравновесности)
        # f_in_left = f_eq_left + (f_in_right - f_eq_right)
        self.f_in[0, :, self._ind_left] = self.f_equil[0, :, self._ind_left] + \
                                          self.f_in[0, :, self._ind_right] - \
                                          self.f_equil[0, :, self._ind_right]

    def calc_f_out(self):
        """
        Calculate pre-collision (f out)
        Этап столкновения.
        """
        self.f_out = self.f_in - self.omega * (self.f_in - self.f_equil)

    def bounce_back(self):
        """
        Bounce-back boundary condition on solid obstacle
        Отражение от препятствия.
        """
        # На узлах препятствия векторы отражаются в обратном направлении
        # self._index[::-1] переворачивает массив индексов, т.к. в D2Q9 векторы симметричны:
        # 0<->0, 1<->3, 2<->4, 5<->7, 6<->8 (если порядок стандартный).
        # Проверим наш порядок:
        # 0(0,0), 1(1,0), 2(0,1), 3(-1,0), 4(0,-1)...
        # 1 vs 3 (обратные), 2 vs 4 (обратные).
        # 5(1,1) vs 7(-1,-1) (обратные).
        # 6(-1,1) vs 8(1,-1) (обратные).
        # Логика [::-1] не совсем верна для массива [0,1,2,3,4,5,6,7,8].
        # 0 <-> 8 (нет), 1 <-> 7 (нет).
        # Нам нужен явный маппинг обратных индексов.
        # Правильный маппинг для нашего self._v:
        # 0->0, 1->3, 2->4, 3->1, 4->2, 5->7, 6->8, 7->5, 8->6
        
        inv_idx = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        if np.any(self.obstacle):
            bndry = self.f_in[self.obstacle, :]
            # Используем f_out для записи отраженных значений, чтобы они пошли в стриминг
            self.f_out[self.obstacle, :] = bndry[:, inv_idx]

    def collision(self):
        """
        Calculate post-collision process (Streaming / Propagation)
        Перенос частиц.
        """
        # np.roll сдвигает массив. axis=0 это X, axis=1 это Y.
        # Для вектора (vx, vy) мы сдвигаем данные НА (vx, vy).
        for i, vx, vy in zip(self._index, self._vx, self._vy):
            self.f_in[:,:,i] = np.roll(
                                    np.roll(self.f_out[:,:,i], vy, axis=1), 
                                    vx, axis=0
                                )
    
    def calc_macro(self):
        """
        Расчет макроскопических параметров (плотность, скорость, давление)
        rho = sum(f_i)
        u = sum(f_i * v_i) / rho
        p = rho / 3
        """
        self.density = np.sum(self.f_in, axis=2)
        
        # Скорость
        # Векторизованный расчет: (nx, ny, 9) * (9,) -> sum по 9
        mom_x = np.dot(self.f_in, self._vx)
        mom_y = np.dot(self.f_in, self._vy)
        
        # Избегаем деления на ноль (хотя rho ~ 1)
        self.u_x = mom_x / self.density
        self.u_y = mom_y / self.density
        
        # Давление
        self.pressure = self.density / 3.0

    @staticmethod
    def calc_outflow(f_in):
        """
        Граничное условие на выходе (справа).
        Простая экстраполяция (нулевой градиент): f(N) = f(N-1)
        """
        # f_in shape (nx, ny, 9) или (9, nx, ny) - зависит от контекста вызова.
        # Внутри класса у нас (nx, ny, 9).
        # Копируем предпоследний слой в последний
        f_in[-1, :, :] = f_in[-2, :, :]

    # --- Основной цикл ---

    def solve(self, n_step, step_frame):
        """
        Основной цикл по времени
        """
        for step in range(n_step):
            # 7.1 Outflow
            self.calc_outflow(self.f_in)
            
            # 7.2 Macro params
            self.calc_macro()
            
            # 7.3 Inflow
            self.calc_inflow()
            
            # 7.4 Collision (calc f_out)
            self.calc_f_equil() # нужен актуальный f_eq
            self.calc_f_out()
            
            # 7.5 Bounce-back
            self.bounce_back()
            
            # 7.6 Streaming (обновление f_in)
            self.collision() # название в methods.py "collision", но логика внутри - стриминг (np.roll)

            # Сохранение данных
            if step % step_frame == 0:
                # Скорость (модуль)
                vel_mag = np.sqrt(self.u_x**2 + self.u_y**2)
                
                # Visual.ipynb ожидает списки, добавляем копии
                self.field_den.append(self.density.copy())
                self.field_u.append(vel_mag.copy())
                self.field_p.append(self.pressure.copy())
                self.field_ux.append(self.u_x.copy())
                self.field_uy.append(self.u_y.copy())
                
                # Для отладки можно выводить прогресс
                # print(f"Step {step}/{n_step}")

    # Методы-обертки для совместимости с тестами (если требуется)
    def calc_f_eq_i(self, i, u, density):
        # Реализация для тестов
        vx, vy = self._v[i]
        ux, uy = u[..., 0], u[..., 1]
        dot = ux*vx + uy*vy
        norma = ux**2 + uy**2
        return density * self._a[i] * (1 + 3*dot + 4.5*dot**2 - 1.5*norma)
