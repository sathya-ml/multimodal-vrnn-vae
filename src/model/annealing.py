from typing import Generator


class AnnealingBetaGenerator(object):
    """
    Generates annealing betas for the annealing process.
    Look at https://arxiv.org/pdf/1903.10145 for more details.
    """

    @staticmethod
    def linear_annealing_beta_generator(
        min_beta: float, max_beta: float, num_epochs: int
    ) -> Generator[float, None, None]:
        for epoch in range(num_epochs):
            proportion: float = (epoch + 1) / num_epochs
            yield min_beta + proportion * (max_beta - min_beta)

    @staticmethod
    def cyclical_annealing_beta_generator(
        num_iterations: int,
        min_beta: float,
        max_beta: float,
        num_cycles: int,
        annealing_percentage: float,
    ) -> Generator[float, None, None]:
        period = int(num_iterations / num_cycles)
        # Linear schedule
        step = (max_beta - min_beta) / (period * annealing_percentage)

        beta = min_beta
        for idx in range(num_iterations):
            yield min(beta, max_beta)

            if idx % period == 0:
                beta = min_beta
            else:
                beta += step

    @staticmethod
    def static_annealing_beta_generator(beta: float) -> Generator[float, None, None]:
        while True:
            yield beta


def _test_annealing_beta_generators():
    print("Testing Linear Annealing Beta Generator:")
    linear_gen = AnnealingBetaGenerator.linear_annealing_beta_generator(
        min_beta=0.0, max_beta=1.0, num_epochs=5
    )
    for beta in linear_gen:
        print(f"Beta: {beta:.2f}")

    print("\nTesting Cyclical Annealing Beta Generator:")
    cyclical_gen = AnnealingBetaGenerator.cyclical_annealing_beta_generator(
        num_iterations=10,
        min_beta=0.0,
        max_beta=1.0,
        num_cycles=2,
        annealing_percentage=0.5,
    )
    for beta in cyclical_gen:
        print(f"Beta: {beta:.2f}")

    print("\nTesting Static Annealing Beta Generator:")
    static_gen = AnnealingBetaGenerator.static_annealing_beta_generator(0.5)
    for _ in range(5):
        print(f"Beta: {next(static_gen):.2f}")


if __name__ == "__main__":
    _test_annealing_beta_generators()
