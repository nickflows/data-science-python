# Introduction to Portfolio Construction w/ Python

## Week 1 - Class Notes

### Fundamentals of Return

+ Average returns are not a good way to look at how assets behave because of volatility
+ Just because of the average return is the same doesn't mean you'll end up with the same amount of money
+ So, how do we characterize returns?

#### From Prices to Returns

+ Return based on price over time. This is called the "Price Return"
$' R_{t,t+1} = \frac{P_{t+1} - P_t}{P_t} '$

+ Make sure to act dividends back at each interval (if there are dividends)
+ This is called the "Total Return". You almost always want to use Total Return

$' R_{t,t+1} = \frac{P_{t+1} + D_{t,t+1}}{P_t} - 1 = \frac{P_{t+1} + D_{t,t+1} - P_t}{P_t} '$


#### Multi-Period Returns

+ Think about compounding the returns over time

$' R_{t,t+2} = \left(1 + R_{t,t+1}\right)\left(1 + R_{t+1,t+2}\right) - 1 '$


#### Annualizing Returns
+ How do you compare returns across different periods of time? By computing annualized returns.
+ Annualized returns is the returns you would get if the rate of return continues for a year (i.e., spread out over a year).
+ Take the per period return and take the product for the number of periods


### Measures of Risk and Reward


#### Volatility - Standard Deviation and Variance
+ A volatile series will deviate off of the mean more significantly


Variance:
+ Variance is the average of the square of the returns
$' \sigma_R^2 = \frac{1}{N} \sum_{i=1}^{N} \left( R_i - \bar{R} \right)^2 '$

Standard Deviation:
+ Standard Deviation is the square root of the variance --> easier to interpret for understanding prices changes in the market

$' \sigma_R = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( R_i - \bar{R} \right)^2} '$

+ Note: We cannot compare the variation of daily data to monthly data and vice versa

Annualized Volatility:
$' \sigma_{\text{ann}} = \sigma_p \sqrt{p} '$
+ p is the number of periods
+ sigma is the variance



#### Risk-Adjusted Measures

+ Return on Risk Ratio = Return / Volatility
+ We should look at its **excess return** over the risk free rate
+ Excesss Return = Return over the Risk Free Rate = Return - Risk Free Rate


**Sharpe Ratio:** (Return - Risk Free Rate) / Volatility

$' \text{Sharpe Ratio}(P) = \frac{R_p - R_f}{\sigma_p} '$


#### Max Drawdown

+ Max Drawdown focuses on the downside risk (instead of the volatility, which measures both upside and downside)
	+ Risk as the possibility of losing money
+ Max Drawdown is the maximum loss you could have had. Buying at the peak and selling at the bottom over some time period (i.e., bought high; sold low)
+ 







