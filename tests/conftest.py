"""Shared pytest command-line options for explicit golden updates."""


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate checked-in deterministic policy snapshots",
    )
