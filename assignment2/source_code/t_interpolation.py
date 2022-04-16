import numpy as np
import matplotlib.pyplot as plt

class Trigono_Interpolation():
    def __init__(self, h):
        self.h = h
        self.x_equation = None
        self.y_equation = None
        self.x_s = None
        self.y_s = None

    def run(self):
        u_s = np.linspace(0, 1, 20)[:-1]
        self.x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*self.h)) + 0.1) * np.cos(12.2*u_s)
        self.y_s = (np.exp(np.sin(6.2*u_s - 0.027*self.h)) + 0.1) * np.sin(12.2*u_s)

        x_data = np.vstack([u_s, self.x_s])
        y_data = np.vstack([u_s, self.y_s])

        self.x_equation = self.interpolate(x_data)
        self.y_equation = self.interpolate(y_data)

    def interpolate(self, points):
        
        u, x = points[0], points[1]
        N = len(u)

        coeff = np.fft.fft(x)
        A = np.array([i.real for i in coeff])
        print('A', A)
        B = np.array([i.imag for i in coeff])
        print('B', B)

        
        u = np.linspace(0, 1, 100)
        mid_point = N//2
        result = A[0]/N + \
            2/N * sum([A[k] * np.cos(2*np.pi*u*k) - B[k] * np.sin(2*np.pi*u*k) for k in range(1, mid_point)]) + \
            A[mid_point]/N * np.cos(np.pi*u*N)

        print(u)
        print(self.x_s)
        print(result)
        print(A[0]/N)
        print(2/N * sum([A[k] * np.cos(2*np.pi*u*k) - B[k] * np.sin(2*np.pi*u*k) for k in range(1, mid_point)]))
        print(A[mid_point]/N * np.cos(np.pi*u*N))

        return result

    def draw(self, path):
        u = np.linspace(0, 1, 100)
        x = 1.5 * (np.exp(1.5 * np.sin(6.2*u - 0.027*self.h)) + 0.1) * np.cos(12.2*u)
        y = (np.exp(np.sin(6.2*u - 0.027*self.h)) + 0.1) * np.sin(12.2*u)

        #plot graph
        fig  = plt.figure()
        plt.title('Trigonometric Interpolation')
        # ax = fig.add_subplot(1,1,1)
        # ax.spines['left'].set_position('center')
        # ax.spines['bottom'].set_position('center')
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')

        plt.plot(self.x_s, self.y_s, 'o', color='grey', label='Points')
        plt.plot(x, y, '--', color='black' ,label='Original Curve')
        plt.plot(self.x_equation, self.y_equation, color='red',label='Fitted Curve')
        plt.legend(loc='best')
        plt.savefig(path)
        plt.show()