

## Week 1: Introduction

### Basic Charting

* Backend is an abstraction layer, which renders matplotlib commands
* Not all backends support all features

### Artist Layers
* Abstraction around drawing and layoyt primatives
* Primatives & Collections
	* Base drawing items
	* Collections end to end in the name "collection"

### Scripting Layers
* Simplify and speed-up interaction w/ environment
* PyPlot is the backend we use for the class


### Procedural 
* PyPlot is a procedural method
* Which drawing actions to take to render darta

### Declarative methods
* Documents of model of relatinships
* DOM or Document Object model
* Example: D3.JS is a declarive method

- Jupyter Notebooks uses the inline backend


## Week 2 - Plotting Libraries

* Imports matplotlib `import matplotlib as mpl`
* Importa matplotlib from pyplot `import matplotlib.pyplot as plt`
* Plots a plot `plt.plot()`


### Scatter Plots
	
- *Principles*
	1.PyPlot keeps track of the axes objects
		- GCF: retrieves current figure
		- GCA: retrieves current axes
	2. PyPlot mirrors the underlying APIs
	3. Function Declaration end with open set of keyword arguments


- Zip Method takes a list of numbers and creates tuples
	- Use list function to show results of iterating over Zip



### Dejunking a Chart (Example Chart)

- Code examples (importing librarties): 
	- import library: `import matplotlib.pyplot as plt`
	- import library: `import numpy as np`
	- plot figure: `plt.figure()`

- change the bar color to be less bright blue:
```
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
```

- make one bar, the python bar, a contrasting color: `bars[0].set_color('#1F77B4')`

- soften all labels by turning grey: `plt.xticks(pos, languages, alpha=0.8)`

- remove the Y label since bars are directly labeled: 
	- `plt.ylabel('% Popularity', alpha=0.8)`
	- `plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)`
	
_ remove all the ticks (both axes), and tick labels on the Y axis: 
```
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
```
	
- remove the frame of the chart: 
	- `for spine in plt.gca().spines.values(): spine.set_visible(False)`
	    
- direct label each bar with Y axis values:
```
for bar in bars:
	plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', ha='center', 	color='w', fontsize=11)

plt.show()
```

## Week 3 - Lessons

- SubPlots
	- Mailing list for developers and users (most focus on users)
	- `plt.subplot(<rows>,<columns>,<currentaxis>)`

- create a 3x3 grid of subplots:
`fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)`

- plot the linear_data on the 5th subplot axes 
`ax5.plot(linear_data, '-')`


## Week 4 - Plotting w/ Pandas and Seaborn

- *Plotting in Pandas*
	- _plots a dataframe as a line graph_ `df.plot();`
	- _plots a scartter plot. Kind can be set as a number of parameters, including:_  *df.plot('A','B', kind = 'scatter');
	- _options:_ `'line', 'bar', 'hist', 'box', 'scatter','area','pie'`
	
- *Seaborn*









