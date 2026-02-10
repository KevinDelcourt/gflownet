import numpy as np
import pickle

def latin_hypercube_sampling(config, visited):
    n_samples = config.pbo_al_experiment.n_samples_per_al_iteration
    dimensions = config.env.dim_profile

    samples_x_without_duplicates = sample(n_samples, dimensions, visited)
    
    with open(config.proxy.model_path, 'rb') as f:
        model = pickle.load(f)

        return samples_x_without_duplicates, model.predict(samples_x_without_duplicates).tolist()

def sample(n_samples, dimensions, visited):
    

    samples_x_without_duplicates = []
    
    strike = 0
    while len(samples_x_without_duplicates) < n_samples and strike < 20:
        batch_size = min(n_samples, min(dimensions))
        samples = np.zeros((batch_size, len(dimensions)), dtype=int)
        for i in range(len(dimensions)):
            samples[:, i] = np.random.permutation(dimensions[i])[:batch_size]
        for sample in samples:
            if sample.tolist() not in visited["X"] and sample.tolist() not in samples_x_without_duplicates:
                samples_x_without_duplicates.append(sample.tolist())
            else:
                strike += 1
                print(f"Found duplicate sample, resampling... (strike {strike})")
                if strike == 20:
                    print("Maximum resampling attempts reached, proceeding with available unique samples.")
                    break
    
    return samples_x_without_duplicates[:n_samples]

if __name__ == "__main__":
    print("Testing Latin Hypercube Sampling...")
    print(sample(5, [5,10,5,5,5], {"X": []}))
