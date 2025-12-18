Example 1: Given a distribution of points (2D, divided in 3 classes, with provided code to generate them), and given a neural network architecture, use SecML Torch to train the network on the given data. Then, compute and plot a security evaluation of this model with an L-inf PGD attack with at least 5 values of epsilon. Discuss the results.

Example 2: Given a distribution of points (with provided code to generate them), and two neural network architectures trained on these data, compute and plot their security evaluation against an L2 attack. Discuss their results in terms of which could be more robust in this setting and why.

Example 3. Provided with an implementation of a maximum confidence or minimum norm attack (as seen during the laboratories), find and fix the bug (it could be anything inside the code, from a wrong sign of the loss to the missing projection operation).

Example 4. Provided with code for computing adversarial training with FGSM, train a model with and without the defense (the architecture is given, along with data).
Compute and plot security evaluations of both and discuss the results.

Example 5. Provided with a model and the data used to train it, compute a security evaluation with a minimum norm attack and a maximum confidence attack. Compare the results by plotting them and discuss them.

The instructors of the course