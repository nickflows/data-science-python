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


### Fundamentals of Return




