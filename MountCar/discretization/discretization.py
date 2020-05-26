import numpy as np
from MountCar.discretization.visual import visualize_samples

def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    intervals=[(high[0]-low[0])/bins[0],(high[1]-low[1])/bins[1]]
    x_arr=np.arange(low[0],high[0],intervals[0])[1:]
    y_arr=np.arange(low[1],high[1],intervals[1])[1:]
    return [x_arr,y_arr]


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
        连续空间中每个点的坐标（x,y）对应着网格世界里格子的索引（x方向第i个格子，y方向第j个格子）
    """
    discretized = []
    for x, bins in zip(sample, grid):
        discretized.append(np.digitize(x, bins))

    return discretized


if __name__=="__main__":
    # 测试网格的空间的创建
    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    grid=create_uniform_grid(low, high)
    print(grid)

    # 测试离散化函数
    grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
    samples = np.array(
        [[-1.0, -5.0],
         [-0.81, -4.1],
         [-0.8, -4.0],
         [-0.5, 0.0],
         [0.2, -1.9],
         [0.8, 4.0],
         [0.81, 4.1],
         [1.0, 5.0]])
    discretized_samples = np.array([discretize(sample, grid) for sample in samples])
    print("\nSamples:", repr(samples), sep="\n")
    print("\nDiscretized samples:", repr(discretized_samples), sep="\n")

    # 测试可视化函数
    visualize_samples(samples, discretized_samples, grid, low, high)








