import numpy as np

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    pass


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    # TODO: Implement this
    pass


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
    """
    # TODO: Implement this
    pass


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    # TODO: Implement this
    pass


if __name__=="__main__":
    low = [-1.0, -5.0]
    high = [1.0, 5.0]
    create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5))  # [test]

    # Tiling specs: [(<bins>, <offsets>), ...]
    tiling_specs = [((10, 10), (-0.066, -0.33)),
                    ((10, 10), (0.0, 0.0)),
                    ((10, 10), (0.066, 0.33))]
    tilings = create_tilings(low, high, tiling_specs)

    # Test with some sample values
    samples = [(-1.2, -5.1),
               (-0.75, 3.25),
               (-0.5, 0.0),
               (0.25, -1.9),
               (0.15, -1.75),
               (0.75, 2.5),
               (0.7, -3.7),
               (1.0, 5.0)]
    encoded_samples = [tile_encode(sample, tilings) for sample in samples]
    print("\nSamples:", repr(samples), sep="\n")
    print("\nEncoded samples:", repr(encoded_samples), sep="\n")








