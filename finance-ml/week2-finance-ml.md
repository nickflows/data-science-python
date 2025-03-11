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


+ \sigma_{\text{semi}} = \sqrt{\frac{1}{N} \sum_{R_t \leq \bar{R}} (R_t - \bar{R})^2}
	+ where N is the number of returns that fall below the mean


+ Value at Risk (VaR)
	+ Represents the maximum expected loss over a given time period
	+ A specified confidence interval - 99% Value at Risk (VaR) --> worst possible outcome after excluding the 1% extreme losses
	+ Over a specified time period (e.g., 1 month)

+ Conditional Value at Risk (VaR)
	+ Expected Loss Beyond VaR
	+ CVaR = - E(R \mid R \leq -VaR) = \frac{-\int_{-\infty}^{-VaR} x \cdot f_R(x) \,dx}{F_R(-VaR)}
	




