import numpy as np
from sklearn import datasets
from pandas import DataFrame


iris = datasets.load_iris()

iris_frame = DataFrame(iris.data)

iris_frame.columns = iris.feature_names

iris_frame['target'] = iris.target

iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])

df_drop = iris_frame.drop(iris_frame.columns[[-1,-2]], axis = 1)

# print(df_drop)


# 2D массив arr_x
arr_x = np.array(df_drop)


# Порахувати mean для 1-ї колонки
means = np.mean(arr_x, axis=0)
col_0_mean = means[0]
# print(col_0_mean)


# Порахувати median для 1-ї колонки
medians = np.median(arr_x, axis=0)
col_0_median = medians[0]
# print(col_0_median)


# Порахувати standard deviation для 1-ї колонки
stdevs = np.std(arr_x, axis=0)
col_0_std = stdevs[0]
# print(col_0_std)


# Вставити 20 значень np.nan на випадкові позиції в масиві
n = 20
index_nan = np.random.choice(arr_x.size, n, replace=False)
arr_x.ravel()[index_nan] = np.nan
# print(arr_x)


# Знайти позиції вставлених значень np.nan в 1-й колонці
positions = np.argwhere(np.isnan(arr_x))
nan_locate = positions.tolist()
nan_index = [i[0] for i in nan_locate if i[1] == 0]
# print(nan_index)


# Відфільтрувати массив за умовою: значення в 3-й колонці > 1.5 та значения в 1-й колонці < 5.0
filt_arr = arr_x[
    ((arr_x[:, 0] < 5.0) & (arr_x[:, 2] > 1.5))
    | ((arr_x[:, 0] < 5.0) & (np.isnan (arr_x[:, 2])))
    | ((np.isnan (arr_x[:, 0])) & (arr_x[:, 2] > 1.5))
]
# print(filt_arr)


# Замінити всі значення np.nan на 0
filt_arr[np.isnan (filt_arr)] = 0
# print(filt_arr)


# Порахувати всі унікальні значення в массиві та вивести їх разом із кількістю
# print(f"Number of unique values: {len (np.unique (filt_arr))}\nUnique values: {np.unique (filt_arr)}")


# Розбити масив по горизонталі на 2 рівні частини
[A,B] = np.hsplit(filt_arr, 2)


# Відсортувати обидва массиви по 1-й колонці (відсортував за 2 колонкою): 1-й за збільшенням, 2-й за зменшенням
arr_sort_01 = A[np.argsort(A[:, 1])]
arr_sort_02 = B[np.argsort(B[:, 1])[::-1]]


# Зібрати обидва массиви в одне ціле
union_arr = np.hstack((arr_sort_01,arr_sort_02))
# print(union_arr)


# Знайти найбільш часто повторюване значення в массиві
vals, counts = np.unique(union_arr, return_counts=True)
# print(vals[np.argmax(counts)])


# Написати функцію, яка б множила всі значення в колонці, які менше середнього значения в цій колонці, на 2, і ділила інші значення на 4
# Застосувати отриману функцію до 3-ї колонки
def arith_mean(x):

    curr_col = union_arr[:, x]

    print(np.mean(curr_col))

    new_column_01 = np.where(curr_col < np.mean(curr_col), curr_col * 2, curr_col) 

    new_column_02 = np.where(curr_col > np.mean(curr_col), curr_col / 4, new_column_01)

    union_arr[:, x] = new_column_02

    return union_arr

print(arith_mean(2))
