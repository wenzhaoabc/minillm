"""Backward-compatible entrypoint for the main SFT trainer."""

import runpy


if __name__ == "__main__":
    runpy.run_module("minillm.train.sft", run_name="__main__")
