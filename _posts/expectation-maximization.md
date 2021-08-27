---
title: "Expectation-Maximization"
date: 2020-07-15
author: Joao Gomes
category: Data Science
cover: 
tags: data science
summary: We explain the theory of the expectation-maximization algorithm.
---

Often we have to deal with hidden variables in machine learning problems. The maximum-likelihood algorithm requires "integrating" over these hidden variables if we want to compare with the observed distribution. However this can lead to a serious problem since we have to deal with sums inside the logarithms. That is, we are instructed to maximize the log-likelihood quantity
$$\sum_i\ln p(x_i)=\sum_i\ln\Big( \sum_h p(x_i,h)\Big)$$
where $h$ is the hidden variable and $x_i$ is the observed one. Except for simple problems, having two sums turns the problem computationally infeasible, especially if the hidden variable is continuous. To deal with this issue we use the concavity property of the logarithm to approximate
$$\ln\Big( \sum_h p(x_i,h)\Big)\geq \sum_hq(h)\ln\Big(\frac{p(x_i,h)}{q(h)}\Big)$$
where $q(h)$ is an unknown distribution that we will want to fix. Further we write
$$\ln p(x_i)=\sum_hq(h)\ln\Big(\frac{p(x_i,h)}{q(h)}\Big)+R_i$$
where the remaining $R_i$ is given by
$$R_i=-\sum_h q(h)\ln\Big(\frac{p(h|x_i)}{q(h)}\Big)=KL(p(h|x_i)||q(h))$$
which is the Kullback-Leibler divergence. Since $R_i\geq 0$ by definition, we have that
$$\ln p(x_i|\theta)\geq \langle \ln p(x_i,h|\theta)\rangle_{q(h)}-\langle \ln q(h)\rangle_{q(h)}$$
where we have introduced prior parameters $\theta$, without lack of generality. The lower bound is saturated provided we choose 
$$\text{E-step:}\quad q(h_i)=p(h_i|x_i,\theta_0)$$
This is also known as expectation E-step. Note that we have a distribution $q(h_i)$ for each sample, as it is determined by $x_i,\theta_0$. However, this step does not solve the maximum-likelihood problem because we still have to fix the parameter $\theta$. What we do next is to maximize the lower bound by choosing $\theta$ keeping $q(h)$ fixed, that is,
$$\text{M-step:}\quad \frac{\partial}{\partial \theta}\langle \ln p(x_i,h|\theta)\rangle_{q(h)}=0$$

Lets take an example that can help clarify some of these ideas. Consider the model which is a mixture of two normal distributions:
$$p(x,c)=\phi(x|\mu_c,\sigma_c)\pi_c,\quad c=0,1$$
where $\phi(x|\mu,\sigma)$ is a normal distribution with mean $\mu$ and variance $\sigma$, and $\pi_c=p(c)$ with $\pi_0+\pi_1=1$. In this example $\theta\equiv \mu,\sigma$, and the hidden variable is $h\equiv c$. 

In the E-step we calculate:
$$\text{E-step:}\quad q(h)=p(h|x,\mu_h,\sigma_h)=\frac{\phi(x|\mu_h,\sigma_h)\pi_h}{\sum_c \phi(x|\mu_c,\sigma_c)\pi_c}$$
We write $q(h_i=0)=\gamma_i(x_i)$ and $q(h_i=1)=1-\gamma_i(x_i)$ for each sample $x_i$, with $\gamma$ given by the ratio above. The initial parameters $\mu,\sigma$ are arbitrary.

The maximization step consists in maximizing the lower bound of the log-likelihood, hence
$$\begin{aligned}\text{M-step:}\quad &\gamma\ln p(x,h=0|\mu,\sigma)+(1-\gamma)\ln p(x,h=1|\mu,\sigma)\\
=&\gamma \ln \phi(x|\mu_0,\sigma_0)+(1-\gamma)\ln \phi(x|\mu_1,\sigma_1)-\gamma\frac{1}{2}\ln\sigma_0-(1-\gamma)\frac{1}{2}\ln\sigma_1+\ldots\\
=& -\gamma \frac{(x-\mu_0)^2}{2\sigma_0^2}-(1-\gamma) \frac{(x-\mu_1)^2}{2\sigma_1^2}-\gamma\frac{1}{2}\ln\sigma_0-(1-\gamma)\frac{1}{2}\ln\sigma_1+\ldots\end{aligned}$$
where $\ldots$ do not depend on $\mu,\sigma$. We need to sum over all samples, so the maximum is calculated
$$\mu_0=\frac{\sum_i x_i\gamma_i}{\sum_i \gamma_i},\;\mu_1=\frac{\sum_i x_i(1-\gamma_i)}{\sum_i (1-\gamma_i)}$$
and 
$$\sigma_0=\frac{\sum_i\gamma_i(x_i-\mu_0)^2}{\sum_i\gamma_i},\quad \sigma_1=\frac{\sum_i(1-\gamma_i)(x_i-\mu_1)^2}{\sum_i(1-\gamma_i)}$$
Maximizing relatively to the probabilities $\pi$ gives
$$\pi_0=\frac{1}{n}\sum_i\gamma_i,\;\pi_1=1-\pi_0$$