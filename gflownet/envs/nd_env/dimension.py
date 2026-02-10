from typing import Iterable, List
import numpy as np

class Dimension:
    
    def __init__(
        self,
        name: str,
        values: Iterable
    ):
        self.name = name
        self.values = list(values)
        self.cells = np.linspace(-1, 1, len(self.values))

    def __repr__(self):
        return f"Dimension(name={self.name!r}, values={self.values})"

    def contains(self, value) -> bool:
        return value in self.values

def make_dims(sizes: List[int]) -> List[Dimension]:
    return [Dimension(f"Dim_{i}", range(size)) for i, size in enumerate(sizes)]

dim_factories = {
    'demo_dims': lambda: [
        Dimension("Power (W)", range(100,550,50)),
        Dimension("Precursor", ["A", "B", "C", "etc"])
    ]
}
