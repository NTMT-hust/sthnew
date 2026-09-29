"""Leakage-resistant site split requested for the DRIAMS experiment."""

TRAIN_SITES = ("DRIAMS-A",)
VALIDATION_SITES = ("DRIAMS-D", "DRIAMS-C")
TEST_SITES = ("DRIAMS-B",)


def split_sites(sites):
    """Return sample/edge indices by source site."""
    train = [i for i, site in enumerate(sites) if site in TRAIN_SITES]
    validation = [i for i, site in enumerate(sites) if site in VALIDATION_SITES]
    test = [i for i, site in enumerate(sites) if site in TEST_SITES]
    return train, validation, test
