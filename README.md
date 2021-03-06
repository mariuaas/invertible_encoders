# Invertible and Pseudo-Invertible Encoders: An Approach to Inverse Problems with Neural Networks

This is the GitHub depository for my Master Thesis in Data Science with special focus on Statistics and Machine Learning at the University of Oslo, Autumn 2021.

The repository includes code and notebooks for reproducing the result we report in the thesis. We note that the seeding of the random number generators can vary in different environments.

*Note that some of the computations are computationally heavy, in particular the Jacobian computations from the first experiments. We stress that running these requires a lot of available memory.*


### Abstract

While neural networks have been demonstrated to be highly successful in mathematical and statistical modelling of a comprehensive selection of problems, their application to inverse problems is not without complications. Recent works have shown that neural networks are especially prone to stability issues -- both in a classical sense, and in the context of so called adverserial attacks which has come to be regarded as the most pervasive source of instability in modern neural network models.

Concomitantly, methods of constructing invertible neural networks with diffeomorphic layer structures with normalizing flows have been proposed as an interesting method for approaching inverse problems by probabilistic augmentation of latent variable models to induce full-rank in a conditional setting. However, these models can often be prohibitively expensive in terms of computational efficiency and memory usage while displaying sufficiently different architectures as to not be trivially extendable to tools for commonly defined feed-forward neural networks.

In this thesis, we motivate the theory of inverse problems via integral equations and spectral theory and discuss the connection of statistical learning theory to neural networks with special focus on encoder-decoder models. Furthermore, in the context of neural networks, we will discuss the underlying theory of epistemic and aleatoric uncertainty, discuss the role of probabilistic modelling, and evaluate the idea of latent probabilistic completion as a remedial method for undercomplete modelling tasks.

Our contribution to the subject of invertible neural networks can be summarized as follows. We propose a relatively simple architectural modification of existing encoder-decoder models using both implicit and explicit orthogonal constraints using Riemannian manifold learning and resolvent operators to construct a wider class of invertible neural networks which are compatible with classic feed-forward architectures. We show that these models provide both a significant decrease in the parameter space compared to standard encoder-decoder networks, as well as theoretical guarantees of robustness and stability without significant loss to model performance. We apply these architectures in combination with existing variational Bayesian methods for a generative approach to underdetermined inverse problems. To this end, we introduce a class of piecewise diffeomorphic activation functions and a bijective Gaussian to Dirichlet transformation for latent variables as an alternative to the canonical Softmax transformation, and propose the application of simple conditional additive coupling layers to improve conditioning in generative models.
