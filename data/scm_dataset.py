"""File containing the SCM dataset logic."""
from scm import SCM


class SCMDataset:
    """Dataset class loading batched data from one random SCM."""
    def __init__(self, num_x: int, num_y: int, batch_size: int):
        """Initializes a new SCM."""
        self.scm = SCM(num_x, num_y)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.scm.generate_data(size=self.batch_size)
