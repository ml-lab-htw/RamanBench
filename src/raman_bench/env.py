"""Load credentials (TABPFN_TOKEN, HUGGING_FACE_HUB_TOKEN, etc.) from a .env file."""

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Load the first .env found from the working directory upward.

    Entry points call this; importing ``raman_bench`` does not. Variables already set in the
    environment win.
    """
    load_dotenv(find_dotenv(usecwd=True))
