(This is an experimental mini-project, work in progress)

Consider image classification task. Let's start with the uniform prior over classes.
Let's introduce a HyperNet that takes the prior belief as the input and outputs target CNN parameters.
After each glance at the input image the probability distribution estimate over target classes is updated and the process is repeated (for a fixed number of steps or until some criterion of convergence / stable oscillation is met).

The idea is to simulate adaptive top-down processing: iterative refining of cognitive hypotheses.
