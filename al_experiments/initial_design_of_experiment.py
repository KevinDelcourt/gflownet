import math
import numpy as np

def generate_initial_data(dim_profiles, n_samples):
    """
        Initial data generator imitating a standard design 
        of experiment approach covering the input 
        space as uniformly as possible.
    """
    design = find_doe(dim_profiles, n_samples)

    options = [list(np.linspace(0, dim - 1, d).astype(int)) for dim, d in zip(dim_profiles, design)]

    all_combinations = np.array(np.meshgrid(*options)).T.reshape(-1, len(dim_profiles))

    

    return all_combinations[:n_samples].tolist()
    

def find_doe(dim_profiles, n_samples):        
    design = [1] * len(dim_profiles)
    i = 0
    while math.prod(design) < n_samples and any(design[j] < dim_profiles[j] for j in range(len(dim_profiles))):
        if design[i] < dim_profiles[i]:
            design[i] += 1
        i = (i + 1) % len(dim_profiles)

    return design
    

if __name__ == "__main__":
    dim_profiles = [8, 32, 5, 2, 2]
    n_samples = 100
    print(find_doe(dim_profiles, n_samples))
    print(generate_initial_data(dim_profiles, n_samples))