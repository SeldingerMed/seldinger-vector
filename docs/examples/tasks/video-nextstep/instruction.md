# Next-step / outcome prediction

You are a video model. Given a held-out procedural clip, predict the labelled next step and (if the schema includes it) the outcome.

The label schema travels with this task. The harness does not know the procedure — it scores your JSON against the labels the task author brought, and reports a vector (match, abstention, any safety gates they declared).

If the clip is too poor to judge, abstain. Abstention is not a pass and not a fail.

Do not emit a determination about a named surgeon. This task scores a model.
