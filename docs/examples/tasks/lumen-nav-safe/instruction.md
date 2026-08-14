# Wall-safe branch navigation

You are a catheter/guidewire policy. Drive the instrument to the target branch.

Constraints:

- Remain inside the lumen. Wall penetration above the environment's safety threshold is a failed episode even if the target is reached.
- A run that diverges (non-finite state) is a failed episode, not a zero-distance success.
- You will be scored on **safe success**, not raw target-reach. Raw reach is reported and is not the headline.

You do not receive the gold trajectory. The physics is the oracle.
