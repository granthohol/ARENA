# %%
import os
import sys
import math
import numpy as np
import einops
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part0_prereqs"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part0_prereqs.utils import display_array_as_img
import part0_prereqs.tests as tests

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")

# %%
display_array_as_img(arr[0])

# %% [markdown]
# # Beginning Exercises

# %%
### Exercise 1
arr1 = einops.rearrange(arr, 'b c h w -> c h (b w)')
display_array_as_img(arr1)

# %%
### Exercise 2
arr2 = einops.repeat(arr[0], 'c h w -> c (2 h) w')
display_array_as_img(arr2)
# %%
### Exercise 3
arr3 = einops.repeat(arr[0:2], 'b c h w -> c (b h) (2 w)') 
display_array_as_img(arr3)
# %%
### Exercise 4 
arr4 = einops.repeat(arr[0], 'c h w -> c (h 2) w')
display_array_as_img(arr4)

# %%
### Exercise 5
arr5 = einops.rearrange(arr[0], 'c h w -> h (c w)')
display_array_as_img(arr5)

# %%
### Exercise 6
arr6 = einops.rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2)
display_array_as_img(arr6)

# %%
### Exercise 7
arr7 = einops.reduce(arr, 'b c h w -> h (b w)', 'max')
display_array_as_img(arr7)

# %%
### Exercise 8
arr8 = einops.reduce(arr, 'b c h w -> h w', 'min' )
display_array_as_img(arr8)
# %%
### Exercise 9
arr9 = einops.rearrange(arr[1], 'c h w -> c w h')
display_array_as_img(arr9)
# %%
### Exercise 10
arr10 = einops.reduce(arr, '(b1 b2) c (h h2) (w w2) ' +
                      '-> c (b1 h) (b2 w)', "max", h2=2, w2=2, b1=2)
display_array_as_img(arr10)
# %% [markdown]
# # Operations
# Defining some testing functions

# %%
def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")

def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")

# %% [markdown]
# Exercises

# %%
### Exercise A.1 - rearrange (1)

def rearrange_1() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    '''


    tens = t.arange(start = 3, end = 9).reshape(3,2)
    print(tens)
    return tens

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

#%%
### Exercise A.2 - rearrange (2)

def rearrange_2() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    '''
    tens = t.arange(1, 7).reshape(2, 3)
    print(tens)
    return tens


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

#%%
def rearrange_3() -> t.Tensor:
    '''Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    '''
    tens = einops.rearrange(t.arange(1, 7), 'a -> 1 a 1')
    print(tens)
    return tens


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))

#%%
## Exercsie B.1
def temperatures_average(temps: t.Tensor) -> t.Tensor:
    '''Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    '''
    assert len(temps) % 7 == 0
    tens = einops.reduce(temps, '(h 7) -> h', 'mean')
    print(tens)
    return tens



temps = t.Tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83])
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)

#%%
## B.2 
def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    '''For each day, subtract the average for the week the day belongs to.

    temps: as above
    '''
    assert len(temps) % 7 == 0
    avg = einops.reduce(temps, '(h 7) -> h', 'mean')
    for i in range(temps.size(0)):
        temps[i] = temps[i] - avg[i // 7]

    return temps


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)

#%%
def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    '''For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    '''
    avg = einops.reduce(temps, '(h 7) -> h', 'mean')
    std = einops.reduce(temps, '(h 7) -> h', t.std)
    for i in range(temps.size(0)):
        temps[i] = (temps[i] - avg[i // 7])/std[i // 7]

    return temps


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)

#%%
## Exercise D
def sample_distribution(probs: t.Tensor, n: int) -> t.Tensor:
    '''Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use torch.rand and torch.cumsum to do this without any explicit loops.

    Note: if you think your solution is correct but the test is failing, try increasing the value of n.
    '''
    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()
    return (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)




n = 10000000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs, rtol=0.001, atol=0.001)
