## Hypotheses

### Hypothesis 1:
All layers work on roughly the same space, that is, their inputs and outputs are tensors from the same distribution. If that is true, we can change the order of the layers and they might still make sense. We can also skip some layers and that might make sense.

### Hypothesis 2:
Not all layers are used on all inputs. In other words, there are inputs for which we can skip some of the layers, and the output will not change by much. This is supported by the "circuits" theory where on some tasks you can find a circuit inside the transformer that is made out of a subset of the transformer layers.
