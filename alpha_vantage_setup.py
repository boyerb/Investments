"""Utilities for loading the Alpha Vantage API key securely."""

import os


def get_alpha_vantage_api_key() -> str:
    """Return the Alpha Vantage key from Colab Secrets or the local environment."""

    try:
        # In Colab, retrieve the key from the Secrets panel.
        from google.colab import userdata

        api_key = userdata.get("ALPHAVANTAGE_API_KEY")
    except ImportError:
        # Local Jupyter and other environments use the operating-system variable.
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY is not available. Add it to Colab Secrets "
            "or set it in Windows, then restart Jupyter."
        )

    print("Alpha Vantage API key loaded.")
    return api_key
