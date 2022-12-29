import numpy as np

LINE_SIZE = 300
COLUMN_SIZE = 6
RANDOM_MAX_RANGE = 10
RANDOM_MIN_RANGE = -10


def array_negatively_correlated(nums):
    return np.sort(nums)[::-1]


def array_positively_correlated(nums):
    return np.sort(nums)


def array_close_to_mean(size, target_mean):
    nums = [target_mean * round(np.random.uniform(RANDOM_MIN_RANGE, RANDOM_MAX_RANGE), 2) for _ in range(size)]
    mean = np.mean(nums)
    nums = [x - mean + target_mean for x in nums]
    return nums


def get_line_array(i, size, global_size):
    if i % int(global_size / 3) == 0:
        # line with a mean close to 2.5
        return array_close_to_mean(size, 2.5)
    nums = []
    if i % 3 == 0:
        nums = [int(np.random.uniform(RANDOM_MIN_RANGE, RANDOM_MAX_RANGE)) for _ in range(size)]
    else:
        nums = [round(np.random.uniform(RANDOM_MIN_RANGE, RANDOM_MAX_RANGE), 2) for _ in range(size)]
    if i % 10 == 0:
        nums = array_positively_correlated(nums)
    if i % 20 == 0:
        nums = array_negatively_correlated(nums)
    return nums


data = np.array([get_line_array(i, COLUMN_SIZE, LINE_SIZE) for i in range(LINE_SIZE)])

np.save('data.npy', data)

t = np.load('data.npy')
print('SHAPE')
print(np.shape(t))
print('\nPOSITIVELY CORRELATED')
print(t[10])
print('\nNEGATIVELY CORRELATED')
print(t[20])
print('\nCLOSE TO 2.5')
print(np.mean(t[100]))
