from ioh import get_problem, ProblemClass
import numpy as np

# pseudo bayesian optimization
def generate_random_data(pbo_problem, dim_profile, n_samples, random_state=None):
    """ 
        Generate random data for PBO problems.
        The input is generated as random integers, 
        which are then converted to binary lists 
        according to the dimension profile. This allows for 
        more efficient sampling in cases where the dimension 
        profile has large values, while still providing the 
        necessary binary input for the PBO problem.
    """


    rng = np.random.default_rng(random_state)
    X = rng.integers(low=0, high=dim_profile, size=(n_samples, len(dim_profile)))
    y = [pbo_problem(integer_list_to_binary_list(x, dim_profile)) for x in X]
    return X, y

def integer_list_to_binary_list(x, dim_profile):
    binary_list = []
    for fragment, width in zip(x, dim_profile):
        binary_list.extend(np.binary_repr(fragment, width=int(np.log2(width))))  
    return [int(bit) for bit in binary_list]

if __name__ == "__main__":
    pbo_problem = get_problem("OneMax", 1, 10, ProblemClass.PBO)
    X, y = generate_random_data(pbo_problem, [8, 32, 2, 2], 5)
    for x in X:
        print(x)
        print(integer_list_to_binary_list(x, [8, 32, 2, 2]))
    print(y)

    print(pbo_problem.meta_data.n_variables)
    print(pbo_problem.meta_data.instance)
    print(pbo_problem.meta_data.name)