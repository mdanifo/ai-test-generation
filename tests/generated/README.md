# Accepted generated tests

Tests here were written by the generator, passed the gate (the clean build twice, once with the other proposals and once on its own, then an assertion failed under at least one seeded bug), and were accepted by a person with `taskgen accept <card>`, which is the only thing that writes here.

Each file is `<card>/<test name>.py` and opens with a header naming the card, the criterion and the seeded bugs the test caught. The code under the header is exactly what the gate ran, so ruff does not reformat this folder; `tests/taskgen/test_generated_suite.py` checks every file still has its header and still passes the static checks.

Empty until accepted tests are committed.
