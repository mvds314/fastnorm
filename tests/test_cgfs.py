#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest


def test_bvn():
    pass


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_bvn",
                "--tb=auto",
                "--pdb",
            ]
        )
