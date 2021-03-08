import numpy as np
import matplotlib.pyplot as plt
import numpy as np




# def show(ori_func, ft, sampling_period=5):
#     n = len(ori_func)
#     interval = sampling_period / n
#     plt.subplot(2, 1, 1)
#     plt.plot(np.arange(0, sampling_period, interval), ori_func, 'black')
#     plt.xlabel('Time'), plt.ylabel('Amplitude')
#     plt.subplot(2, 1, 2)
#     frequency = np.arange(n / 2) / (n * interval)
#     nfft = abs(ft[range(int(n / 2))] / n)
#     plt.plot(frequency, nfft, 'red')
#     plt.xlabel('Freq (Hz)'), plt.ylabel('Amp. Spectrum')
#     plt.show()


from scipy import linalg
from numpy.polynomial import polynomial
# x = np.random.random((100, 100))
# y = x[42, 87]
#
# print(x[0,3:5])

# c_array = np.random.rand(10000, 10000)

# Conversion a arreglos de tipo Fortran
# f_array = np.asfortranarray(c_array)

# x = range(5)
# y = np.array(x)

# x = np.arange(5)
# LOW, HIGH = 1, 11
# SIZE = 10
# x = np.ones((10, 10), dtype=int)
# x = np.array([[ 0, 0, 0], [10,10,10], [20,20,20]])
# y = np.array([1, 2, 3])
#
# list_ex = np.zeros((2,), dtype = [('id', 'i4'), ('value', 'f4', (2,))])
# dict_ex = np.zeros((2,), dtype = {'names':['id', 'value'], 'formats':['i4', '2f4']})

# x = np.array([[4,8],[7,9]])
# A = np.array([[1,2],[3,4]])
# A = np.mat('3 1 4; 1 5 9; 2 6 5')
# b = np.mat([[1],[2],[3]])
#
#
#
# def sum_row(x):
#     """
#     Given an array `x`, return the sum of its zeroth row.
#     """
#     return np.sum(x[0, :])
#
#
# def sum_col(x):
#     """
#     Given an array `x`, return the sum of its zeroth column.
#     """
#     return np.sum(x[:, 0])
#
