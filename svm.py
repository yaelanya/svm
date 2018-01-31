import numpy as np

class SVM(object):
    def __init__(self, C=1.0, tol=0.01, gamma=1.0, eps=0.01):
        self.C = C
        self.tol = tol
        self.eps = eps
        self.gamma = gamma


    def predict(self, X_test):
        self.w = np.dot(self.a * self.target, self.point)
        self.b = np.mean(self.target - np.dot(self.w.T, self.point.T))

        label_list = []
        for x in X_test:
            #predict_value = sum(self.a * self.target * self.kernel(x, self.point)) - self.b
            #predict_value = sum([self.a[i] * self.target[i] * self.kernel(x, self.point[i]) for i in range(self.N)]) - self.b
            predict_value = np.dot(self.w.T, x) + self.b

            if predict_value > 0:
                label_list.append(1)
            else:
                label_list.append(0)

        return np.array(label_list)


    def kernel(self, x1, x2):
        '''
        RBF kernel
        '''
        tmp = sum([np.linalg.norm(x1[i] - x2[i]) for i in range(np.ndim(x1))])

        return np.exp(-self.gamma * tmp)


    def take_step(self, i1, i2):
        if i1 is i2:
            return 0

        alpha1 = self.a[i1]
        alpha2 = self.a[i2]
        y1 = self.target[i1]
        y2 = self.target[i2]
        E1 = self.E[i1]
        E2 = self.E[i2]
        s = y1 * y2

        # Compute L, H
        if y1 is not y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L is H:
            return 0

        k11 = self.kernel(self.point[i1], self.point[i1])
        k12 = self.kernel(self.point[i1], self.point[i2])
        k22 = self.kernel(self.point[i2], self.point[i2])

        eta = 2 * k12 - k11 - k22
        if eta < 0:
            a2 = alpha2 - y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            alpha_adj = self.a.copy()
            alpha_adj[i2] = L

            L_obj = sum(alpha_adj) - 0.5 * sum(self.target * self.target * self.kernel(self.point, self.point) * alpha_adj * alpha_adj)
            alpha_adj[i2] = H
            H_obj = sum(alpha_adj) - 0.5 * sum(self.target * self.target * self.kernel(self.point, self.point) * alpha_adj * alpha_adj)

            # clipped a2
            if L_obj > H_obj + self.eps:
                a2 = L
            elif L_obj < H_obj - self.eps:
                a2 = H
            else:
                a2 = alpha2

        if a2 < 1e-8:
            a2 = 0
        elif a2 > self.C - 1e-8:
            a2 = self.C

        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return 0

        a1  = alpha1 + s * (alpha2 - a2)

        b_old = self.b
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + b_old
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + b_old

        if a1 > 0 and a1 < self.C:
            self.b = b1
        elif a2 > 0 and a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        self.E = [self.E[k] + y1 * (a1 - self.a[i1]) * self.kernel(self.point[i1], self.point[k]) + \
            y2 * (a2 - self.a[i2]) * self.kernel(self.point[i2], self.point[k]) + b_old - self.b for k in range(self.N)]

        self.a[i1] = a1
        self.a[i2] = a2

        return 1


    def examine_example(self, i2):
        y2 = self.target[i2]
        alpha2 = self.a[i2]
        E2 = self.E[i2]
        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            n_count = sum([1 if (a_i == 0) and (a_i != self.C) else 0 for a_i in self.a])
            if n_count > 1:
                if E2 >= 0:
                    i1 = np.argmin(self.a)
                else:
                    i1 = np.argmax(self.a)

                if self.take_step(i1, i2):
                    return 1

            for i1 in np.roll(np.where((self.a != 0).all() and (self.a != self.C).all())[0], np.random.choice(self.N)):
                if self.take_step(i1, i2):
                    return 1

            for i1 in np.roll(np.arange(self.N), np.random.choice(self.N)):
                if self.take_step(i1, i2):
                    return 1

        return 0


    def fit(self, X, target):
        self.N = len(X)
        self.a = np.zeros(self.N)
        self.point = np.array(X)
        self.target = np.array(target)
        self.E = -self.target.copy()
        self.b = 0

        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.N):
                    num_changed += self.examine_example(i)
            else:
                for i  in np.where((self.a != 0).all() and (self.a != self.C).all())[0]:
                    num_changed += self.examine_example(i)

            if examine_all is 1:
                examine_all = 0
            elif num_changed is 0:
                examine_all = 1