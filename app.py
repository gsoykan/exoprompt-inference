"""HF Spaces / `streamlit run` entry point.

Streamlit imports this file as a module and executes top-level code, so we delegate
to the real app inside the package.
"""

from exoprompt_inference.streamlit.app import main

if __name__ == "__main__" or True:
    main()
