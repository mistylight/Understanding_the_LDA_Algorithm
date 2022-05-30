# Understanding the LDA Algorithm

Example codes for my blog post: [Understanding the LDA Algorithm](https://mistylight.github.io/posts/33996/).

Note: Before running the notebooks, please go to the `data/` folder and generate the training data according to the instructions in the README.

## LDA Training

### Topics extracted for 19xx NeurIPS papers:

![img](https://raw.githubusercontent.com/mistylight/picbed/main/Hexo/2022_05_30_1MS4KjsFfygB6XE.png)

We can interpret the topics as (note that this interpretation is purely subjective):

- Topic 0: training ML models
- Topic 1: math symbols / circuit design
- Topic 2: signal processing
- Topic 3: reinforcement learning
- Topic 4: neural networks
- Topic 5: kernel methods
- Topic 6: speech recognition
- Topic 7: biological inspiration for neural networks
- Topic 8: computer vision
- Topic 9: probability theory / bayesian networks

### Topics extracted for 20xx NeurIPS papers:

![img](https://raw.githubusercontent.com/mistylight/picbed/main/Hexo/2022_05_30_eXLfbapBF9OQo5w.png)

We can interpret the topics as (note that this interpretation is purely subjective):

- Topic 0: training deep neural networks
- Topic 1: general machine learning
- Topic 2: (?)
- Topic 3: generative models (GAN/VAE)
- Topic 4: neural networks
- Topic 5: (?)
- Topic 6: (?)
- Topic 7: reinforcement learning
- Topic 8: graph analysis
- Topic 9: computer vision

## LDA Testing

Extracting topics for a new unseen paper (using the model trained with 19xx NeurIPS papers):

```
Title: Decomposition of Reinforcement Learning for Admission Control
Abstract: This paper presents predictive gain scheduling, a technique for simplifying reinforcement learning problems by decomposition. Link admission
control of self-similar call traffic is used to demonstrate the technique.
The control problem is decomposed into on-line prediction of near-future call arrival rates, and precomputation of policies for Poisson call arrival processes. At decision time, the predictions are used to select among the policies. Simulations show that this technique results in significantly faster learning without any performance loss, compared to a reinforcement learning controller that does not decompose the problem.
```

Result:
```python
[0.1299, 0.0005, 0.0005, 0.6124, 0.0005, 0.0005, 0.2144, 0.0403, 0.0005, 0.0005]
```

The top topic is Topic 3 (reinforcement learning), which aligns with our intuition.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

[@mistylight](https://github.com/mistylight) - mistylight.cs@gmail.com

## Acknowledgements

[1] Latent Dirichlet Allocation. David M. Blei, Andrew Y. Ng, Michael I. Jordan. 2003. https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

[2] Parameter estimation for text analysis. Gregor Heinrich. 2005. http://www.arbylon.net/publications/text-est.pdf

[3] LDA数学八卦. 靳志辉. 2013. https://bloglxm.oss-cn-beijing.aliyuncs.com/lda-LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf
