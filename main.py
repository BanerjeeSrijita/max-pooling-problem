import numpy as np

def max_pooling_naive(image, window_size):
    m = image.shape[0]
    k = window_size

    output = np.zeros((m, m), dtype=int)  

    for i in range(m):
        for j in range(m):
            window = image[i:min(i+k, m), j:min(j+k, m)]
            output[i, j] = np.max(window)

    return output

def max_pooling_optimized(image, window_size):
    m = image.shape[0]
    k = window_size

    output = np.zeros((m, m), dtype=int)  

    max_row = np.zeros((m, m - k + 1), dtype=int)  
    for i in range(m):
        for j in range(m - k + 1):
            max_row[i, j] = np.max(image[i, j:j+k])

    for j in range(m - k + 1):
        for i in range(m):
            output[i, j] = np.max(max_row[i:i+k, j])

    return output

# Test the implementations
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

window_size = 2

print("Naive Max Pooling:")
print(max_pooling_naive(image, window_size))

print("\nOptimized Max Pooling:")
print(max_pooling_optimized(image, window_size))
