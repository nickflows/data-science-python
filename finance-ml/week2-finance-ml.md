# Introduction to Portfolio Construction w/ Python (Week 2)

## Week 2 - Class Notes


### Devitations from Normality
+ Standard simplifying assumption: asset returns are normally distributed
+ Large changes are more common in reality --> normal distribution doesn't capture stock market
+ Higher Order Moments:
	+ Skewness: A measure of symmettry in the distribution
		+ negative skew --> Skews toward below the expected value or mean
		+ positive skew --> Skews toward above the expected value or mean
	+ Kurtosis: A measure of thickness of the tail of the distribution
+ Evidence of Non-Normality: Equity Markets
	+ Asset Returns are not normally distributed


### Downside Risk Measures
+ Volatility is a very symettric measure of risk (standard deviation around the mean)
+ We want to move to more extreme deviation around the mean



+ Semi Deviation: Volatility of the sub-sample of below average or below zero returns


$$ \sigma_{\text{semi}} = \sqrt{\frac{1}{N} \sum_{R_t \leq \bar{R}} (R_t - \bar{R})^2} $$

where N is the number of returns that fall below the mean


+ Value at Risk (VaR)
	+ Represents the maximum expected loss over a given time period
	+ A specified confidence interval - 99% Value at Risk (VaR) --> worst possible outcome after excluding the 1% extreme losses
	+ Over a specified time period (e.g., 1 month)

+ Conditional Value at Risk (VaR)
	+ Expected Loss Beyond VaR

$$ CVaR = - E(R \mid R \leq -VaR) = \frac{-\int_{-\infty}^{-VaR} x \cdot f_R(x) \,dx}{F_R(-VaR)} $$


### Estimating Value at Risk (VaR)
+ Historical Methodology: Calculation of VaR based on the distribution in historical changes in the value of the current portfolio
	+ No assumptions --> estimate might be sensative to the sample period
+ Parametric Gaussian Methodology: Calculation of VaR based on portfolio volatility on volatilities and correlations of components (e.g., guassian or normal distribition)
	+ For Guassian dist., only need to measure the mean and volatility

$$ VaR_{\alpha} = -(\mu + z_{\alpha} \sigma)  $$

+ Parametric Non-Gaussian VaR
	+ An alternative to parametric exists --> semi-parametric approach

+ Corner-Fisher VaR:
	+ An alternative to parametric approaches
	+ Cornish-Fisher Expansion

$$ \tilde{z}_{\alpha} = z_{\alpha} + \frac{1}{6} (z_{\alpha}^2 - 1) S + \frac{1}{24} (z_{\alpha}^3 - 3z_{\alpha}) (K - 3) - \frac{1}{36} (2z_{\alpha}^3 - 5z_{\alpha}) S^2 $$

$$ VaR_{\text{mod}} (1 - \alpha) = -(\mu + \tilde{z}_{\alpha} \sigma) $$


## Week 2 - Introduction to Optimization & Efficient Fronteir

### Mean-Variance Framework
+ Single Period Framework (X-Axis: Risk; Y-Axis: Return)
+ Simple 2-Asset Portfolio
	+ What is the Risk/Return of the Combination of the Two Assets (A&B)
	+ E.g., 50% in A and 50% in B
+ Return on a Portfolio: Weighted Average of the Components of the Portfolio
+ Risk of a Portfolio: Cannot take a Weighted Average of A & B (due to the correlation between A and B)
	+ More de-correlated the assets are, the less the volatility of the combination will be.

+ Portfolio Risk:

$$
\sigma^2(w_a, w_b) = \sigma_A^2 w_A^2 + \sigma_B^2 w_B^2 + 2 w_A w_B \sigma_A \sigma_B \rho_{A,B}
$$


### Markowitz Optimization & Efficient Frontier
+ With 3 assets, you can construst a portfolio over a region of space in (X,Y) plane
+ Efficient Frontier - Want to be on the edge of the Risk/Return region that maximizes return or minimizes risk (when constructing portfolios from some number of assets N)

### Convex Optimization to Draw Efficient Frontier
+ Given a set of Asset Return Volatilities and Correlations, plot the efficient frontier
+ Expressions for the Return & Volatility for a Portfolio

+ Return on Portfolio:

$$
R_p = \sum_{i=1}^{k} w_i R_i
$$

+ Generalized Volatility of N Assets: 

$$
\sigma_p^2 = \sum_{i=1}^{k} \sum_{j=1}^{k} w_i w_j \sigma_i \sigma_j \rho_{ij}
$$


+ Portfolio Variance from Covariances of Assets:

$$
\sigma_p^2 = \sum_{i=1}^{k} \sum_{j=1}^{k} w_i w_j \sigma_{ij}
$$



+ Find the Efficient Frontier (Optimization Problem): 

$$
\text{Minimize: } \frac{1}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}
$$

**Subject to:**

$$
\mathbf{w}^\top \mathbf{R} = r_0
$$

$$
\mathbf{w}^\top \mathbf{1} = 1
$$

$$
\mathbf{w} \geq \mathbf{0}
$$










