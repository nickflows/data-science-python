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












