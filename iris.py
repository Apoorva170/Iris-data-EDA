#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[8]:


#Load iris.csv into a pandas dataFrame.
iris_df = pd.read_csv(r"C:\Users\AISHWARYA\Downloads\iris.csv")
iris_df


# In[9]:


# (Q) how many data-points and features?
print (iris_df.shape)


# In[10]:


#(Q) What are the column names in our dataset?
print (iris_df.columns)


# In[11]:


#(Q) How many data points for each class are present? 

iris_df["species"].value_counts()


# In[12]:


# Checking wether data has any missing values

print(iris_df.info())


# In[13]:


import numpy as np

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"] 


"""To Draw 1-D Scatter Plot we are making x-axis = Feature, Y-axis = zeros """

#1-D scatter plot of sepal_length
plt.figure(1)
plt.plot(versicolor["sepal_length"], np.zeros_like(versicolor['sepal_length']), 'o', label = 'versicolor')
plt.plot(setosa["sepal_length"], np.zeros_like(setosa['sepal_length']), 'o', label = 'setosa')
plt.plot(virginica["sepal_length"], np.zeros_like(virginica['sepal_length']), 'o', label = 'virginica')
plt.title("1-D Scatter plot of sepal_length")
plt.xlabel("sepal_length")
plt.legend()
plt.show()

#1-D scatter plot of sepal_width
plt.figure(1)
plt.plot(versicolor["sepal_width"], np.zeros_like(versicolor['sepal_width']), 'o', label = 'versicolor')
plt.plot(setosa["sepal_width"], np.zeros_like(setosa['sepal_width']), 'o', label = 'setosa')
plt.plot(virginica["sepal_width"], np.zeros_like(virginica['sepal_width']), 'o', label = 'virginica')
plt.title("1-D Scatter plot of sepal_width")
plt.xlabel("sepal_width")
plt.legend()
plt.show()

#1-D scatter plot of petal_length
plt.figure(1)
plt.plot(versicolor["petal_length"], np.zeros_like(versicolor['petal_length']), 'o', label = 'versicolor')
plt.plot(setosa["petal_length"], np.zeros_like(setosa['petal_length']), 'o', label = 'setosa')
plt.plot(virginica["petal_length"], np.zeros_like(virginica['petal_length']), 'o', label = 'virginica')
plt.title("1-D Scatter plot of petal_length")
plt.xlabel("petal_length")
plt.legend()
plt.show()

#1-D scatter plot of petal_width
plt.figure(1)
plt.plot(versicolor["petal_width"], np.zeros_like(versicolor['petal_width']), 'o', label = 'versicolor')
plt.plot(setosa["petal_width"], np.zeros_like(setosa['petal_width']), 'o', label = 'setosa')
plt.plot(virginica["petal_width"], np.zeros_like(virginica['petal_width']), 'o', label = 'virginica')
plt.title("1-D Scatter plot of petal_width")
plt.xlabel("petal_width")
plt.legend()
plt.show()


# In[16]:


# Histogram and PDF for the Independent variable 'sepal_length'

sns.FacetGrid(iris_df, hue="species", height=5)    .map(sns.distplot, "sepal_length")    .add_legend();
plt.title("Histogram of sepal_length")
plt.ylabel("Probability Density of sepal_length")
plt.show()


# In[17]:


# Histogram and PDF for the Independent variable 'sepal_width'

sns.FacetGrid(iris_df, hue="species", height=5)    .map(sns.distplot, "sepal_width")    .add_legend();
plt.title("Histogram of sepal_width")
plt.ylabel("Probability Density of sepal_width")
plt.show()


# In[18]:


# Histogram and PDF for the Independent variable 'petal_length'

sns.FacetGrid(iris_df, hue="species", height=5)    .map(sns.distplot, "petal_length")    .add_legend();
plt.title("Histogram of petal_length")
plt.ylabel("Probability Density of petal_length")
plt.show()


# In[20]:


# Histogram and PDF for the Independent variable 'petal_width'

sns.FacetGrid(iris_df, hue="species", height=5)    .map(sns.distplot, "petal_width")    .add_legend();
plt.title("Histogram of petal_width")
plt.ylabel("Probability Density of petal_width")
plt.show()


# In[21]:


# CDF for the Independent variable 'sepal_length'

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"] 


# CDF of sepal_length for versicolor
plt.figure(1)
counts, bin_edges = np.histogram(versicolor['sepal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')
plt.xlabel("sepal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_length for versicolor")
plt.legend()


# CDF of sepal_length for setosa
plt.figure(2)
counts, bin_edges = np.histogram(setosa['sepal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:], cdf, label = 'cdf of setosa')
plt.xlabel("sepal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_length for setosa")
plt.legend()


# CDF of sepal_length for virginica
plt.figure(3)
counts, bin_edges = np.histogram(virginica['sepal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:], cdf, label = 'cdf of virginica')
plt.xlabel("sepal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_length for virginica")
plt.legend()



# CDFs of versicolor, setosa,virginica for the feature sepal_length in a single plot
plt.figure(4)

counts, bin_edges = np.histogram(versicolor['sepal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')


counts, bin_edges = np.histogram(setosa['sepal_length'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:],cdf, label = 'cdf of setosa')


counts, bin_edges = np.histogram(virginica['sepal_length'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:],cdf, label = 'cdf of virginica')



plt.xlabel("sepal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_length for versicolor,setosa,virginica ")
plt.legend()
plt.show()


# In[22]:


# CDF for the Independent variable 'sepal_width'

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"] 


# CDF of sepal_length for versicolor
plt.figure(1)
counts, bin_edges = np.histogram(versicolor['sepal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')
plt.xlabel("sepal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_width for versicolor")
plt.legend()


# CDF of sepal_length for setosa
plt.figure(2)
counts, bin_edges = np.histogram(setosa['sepal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:], cdf, label = 'cdf of setosa')
plt.xlabel("sepal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_width for setosa")
plt.legend()


# CDF of sepal_length for virginica
plt.figure(3)
counts, bin_edges = np.histogram(virginica['sepal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:], cdf, label = 'cdf of virginica')
plt.xlabel("sepal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_width for virginica")
plt.legend()



# CDFs of versicolor, setosa,virginica for the feature sepal_length in a single plot
plt.figure(4)

counts, bin_edges = np.histogram(versicolor['sepal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')


counts, bin_edges = np.histogram(setosa['sepal_width'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:],cdf, label = 'cdf of setosa')


counts, bin_edges = np.histogram(virginica['sepal_width'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:],cdf, label = 'cdf of virginica')



plt.xlabel("sepal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of sepal_width for versicolor,setosa,virginica ")
plt.legend()
plt.show()


# In[23]:


# CDF for the Independent variable 'petal_length'

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"] 


# CDF of petal_length for versicolor
plt.figure(1)
counts, bin_edges = np.histogram(versicolor['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')
plt.xlabel("petal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_length for versicolor")
plt.legend()


# CDF of petal_length for setosa
plt.figure(2)
counts, bin_edges = np.histogram(setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:], cdf, label = 'cdf of setosa')
plt.xlabel("petal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_length for setosa")
plt.legend()


# CDF of petal_length for virginica
plt.figure(3)
counts, bin_edges = np.histogram(virginica['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:], cdf, label = 'cdf of virginica')
plt.xlabel("petal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_length for virginica")
plt.legend()



# CDFs of versicolor, setosa,virginica for the feature petal_length in a single plot
plt.figure(4)

counts, bin_edges = np.histogram(versicolor['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')


counts, bin_edges = np.histogram(setosa['petal_length'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:],cdf, label = 'cdf of setosa')


counts, bin_edges = np.histogram(virginica['petal_length'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:],cdf, label = 'cdf of virginica')



plt.xlabel("petal_length")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_length for versicolor,setosa,virginica ")
plt.legend()
plt.show()


# In[24]:


# CDF for the Independent variable 'petal_width'

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"] 


# CDF of petal_width for versicolor
plt.figure(1)
counts, bin_edges = np.histogram(versicolor['petal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')
plt.xlabel("petal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_width for versicolor")
plt.legend()


# CDF of petal_width for setosa
plt.figure(2)
counts, bin_edges = np.histogram(setosa['petal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:], cdf, label = 'cdf of setosa')
plt.xlabel("petal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_width for setosa")
plt.legend()


# CDF of petal_length for virginica
plt.figure(3)
counts, bin_edges = np.histogram(virginica['petal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:], cdf, label = 'cdf of virginica')
plt.xlabel("petal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_width for virginica")
plt.legend()



# CDFs of versicolor, setosa,virginica for the feature petal_width in a single plot
plt.figure(4)

counts, bin_edges = np.histogram(versicolor['petal_width'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of versicolor');
plt.plot(bin_edges[1:], cdf, label = 'cdf of versicolor')


counts, bin_edges = np.histogram(setosa['petal_width'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of setosa');
plt.plot(bin_edges[1:],cdf, label = 'cdf of setosa')


counts, bin_edges = np.histogram(virginica['petal_width'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'pdf of virginica');
plt.plot(bin_edges[1:],cdf, label = 'cdf of virginica')



plt.xlabel("petal_width")
plt.ylabel("Cummulative probability density")
plt.title("CDF of petal_width for versicolor,setosa,virginica ")
plt.legend()
plt.show()


# In[26]:


#Mean, Variance, Std-deviation,

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"] 

print('Means:')
print("Mean of sepal_length for versicolor  = ", np.mean(versicolor["sepal_length"]))
print("Mean of sepal_length for setosa = " , np.mean(setosa["sepal_length"]))
print("Mean of sepal_length for virginica  = ", np.mean(virginica["sepal_length"]))
print("***********************************************************************************************")
print("Mean of sepal_width for versicolor  = ", np.mean(versicolor["sepal_width"]))
print("Mean of sepal_width for setosa = " , np.mean(setosa["sepal_width"]))
print("Mean of sepal_width for virginica  = ", np.mean(virginica["sepal_width"]))
print("***********************************************************************************************")
print("Mean of petal_length for versicolor  = ", np.mean(versicolor["petal_length"]))
print("Mean of petal_length for setosa = " , np.mean(setosa["petal_length"]))
print("Mean of petal_length for virginica  = ", np.mean(virginica["petal_length"]))
print("***********************************************************************************************")
print("Mean of petal_width for versicolor  = ", np.mean(versicolor["petal_width"]))
print("Mean of petal_width for setosa = " , np.mean(setosa["petal_width"]))
print("Mean of petal_width for virginica  = ", np.mean(virginica["petal_width"]))


print("\nStd-dev:");
print("Std-dev of sepal_length for versicolor = ", np.std(versicolor["sepal_length"]))
print("Std-dev of sepal_length for setosa = ", np.std(setosa["sepal_length"]))
print("Std-dev of sepal_length for virginica =", np.std(virginica["sepal_length"]))
print("***********************************************************************************************")
print("Std-dev of sepal_width for versicolor = ", np.std(versicolor["sepal_width"]))
print("Std-dev of sepal_width for setosa = ", np.std(setosa["sepal_width"]))
print("Std-dev of sepal_width for virginica =", np.std(virginica["sepal_width"]))
print("***********************************************************************************************")
print("Std-dev of petal_length for versicolor = ", np.std(versicolor["petal_length"]))
print("Std-dev of petal_length for setosa = ", np.std(setosa["petal_length"]))
print("Std-dev of petal_length for virginica =", np.std(virginica["petal_length"]))
print("***********************************************************************************************")
print("Std-dev of petal_width for versicolor = ", np.std(versicolor["petal_width"]))
print("Std-dev of petal_width for setosa = ", np.std(setosa["petal_width"]))
print("Std-dev of petal_width for virginica =", np.std(virginica["petal_width"]))
print("***********************************************************************************************")


# In[28]:


#Median, Quantiles, Percentiles, IQR.

versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"]

print("\nMedians:")
# Median is also a central tendency value like mean, but mean and std can be easily corrupted by outliers. 
#so better to use Median and MAD
print("median of sepal_length for versicolor = ", np.median(versicolor["sepal_length"]))
print("median of sepal_length for setosa = " , np.median(setosa["sepal_length"]))
print("median of sepal_length for virginica = ", np.median(virginica["sepal_length"]))
print("***********************************************************************************************")
print("median of sepal_width for versicolor = ", np.median(versicolor["sepal_width"]))
print("median of sepal_width for setosa = " , np.median(setosa["sepal_width"]))
print("median of sepal_width for virginica = ", np.median(virginica["sepal_width"]))
print("***********************************************************************************************")
print("median of petal_length for versicolor = ", np.median(versicolor["petal_length"]))
print("median of petal_length for setosa = " , np.median(setosa["petal_length"]))
print("median of petal_length for virginica = ", np.median(virginica["petal_length"]))
print("***********************************************************************************************")
print("median of petal_width for versicolor = ", np.median(versicolor["petal_width"]))
print("median of petal_width for setosa = " , np.median(setosa["petal_width"]))
print("median of petal_width for virginica = ", np.median(virginica["petal_width"]))
print("***********************************************************************************************")


print("\nQuantiles:")

print("Quantiles of sepal_length for versicolor = ", np.percentile(versicolor["sepal_length"],np.arange(0, 100, 25)))
print("Quantiles of sepal_length for setosa = " , np.percentile(setosa["sepal_length"],np.arange(0, 100, 25)))
print("Quantiles of sepal_length for virginica = ", np.percentile(virginica["sepal_length"],np.arange(0, 100, 25)))
print("***********************************************************************************************")
print("Quantiles of sepal_width for versicolor = ", np.percentile(versicolor["sepal_width"],np.arange(0, 100, 25)))
print("Quantiles of sepal_width for setosa = " , np.percentile(setosa["sepal_width"],np.arange(0, 100, 25)))
print("Quantiles of sepal_width for virginica = ", np.percentile(virginica["sepal_width"],np.arange(0, 100, 25)))
print("***********************************************************************************************")
print("Quantiles of petal_length for versicolor = ", np.percentile(versicolor["petal_length"],np.arange(0, 100, 25)))
print("Quantiles of petal_length for setosa = " , np.percentile(setosa["petal_length"],np.arange(0, 100, 25)))
print("Quantiles of petal_length for virginica = ", np.percentile(virginica["petal_length"],np.arange(0, 100, 25)))
print("***********************************************************************************************")
print("Quantiles of petal_width for versicolor = ", np.percentile(versicolor["petal_width"],np.arange(0, 100, 25)))
print("Quantiles of petal_width for setosa = " , np.percentile(setosa["petal_width"],np.arange(0, 100, 25)))
print("Quantiles of petal_width for virginica = ", np.percentile(virginica["petal_width"],np.arange(0, 100, 25)))
print("***********************************************************************************************")

print("\n90th Percentiles:")

print("90th Percentiles of sepal_length for versicolor = ", np.percentile(versicolor["sepal_length"],90))
print("90th Percentiles of sepal_length for setosa = " , np.percentile(setosa["sepal_length"],90))
print("90th Percentiles of sepal_length for virginica = ", np.percentile(virginica["sepal_length"],90))
print("***********************************************************************************************")
print("90th Percentiles of sepal_width for versicolor = ", np.percentile(versicolor["sepal_width"],90))
print("90th Percentiles of sepal_width for setosa = " , np.percentile(setosa["sepal_width"],90))
print("90th Percentiles of sepal_width for virginica = ", np.percentile(virginica["sepal_width"],90))
print("***********************************************************************************************")
print("90th Percentiles of petal_length for versicolor = ", np.percentile(versicolor["petal_length"],90))
print("90th Percentiles of petal_length for setosa = " , np.percentile(setosa["petal_length"],90))
print("90th Percentiles of petal_length for virginica = ", np.percentile(virginica["petal_length"],90))
print("***********************************************************************************************")
print("90th Percentiles of petal_width for versicolor = ", np.percentile(versicolor["petal_width"],90))
print("90th Percentiles of petal_width for setosa = " , np.percentile(setosa["petal_width"],90))
print("90th Percentiles of petal_width for virginica = ", np.percentile(virginica["petal_width"],90))
print("***********************************************************************************************")


# In[29]:


# Box plot for the feature 'sepal_length'
# Box plot is drawn for visualizing percentile, quantile.

sns.boxplot(x='species',y='sepal_length', data=iris_df)
plt.title("BOX plot for sepal_length feature")
plt.show()


# In[30]:


# Box plot for the feature 'sepal_length'
# Box plot is drawn for visualizing percentile, quantile.

sns.boxplot(x='species',y='sepal_width', data=iris_df)
plt.title("BOX plot for sepal_width feature")
plt.show()


# In[31]:


# Box plot for the feature 'sepal_length'
# Box plot is drawn for visualizing percentile, quantile.

sns.boxplot(x='species',y='petal_length', data=iris_df)
plt.title("BOX plot for petal_length feature")
plt.show()


# In[32]:


# Box plot for the feature 'sepal_length'
# Box plot is drawn for visualizing percentile, quantile.

sns.boxplot(x='species',y='petal_width', data=iris_df)
plt.title("BOX plot for petal_width feature")
plt.show()


# In[37]:


# 2-D Scatter plot

sns.set_style("whitegrid");
sns.FacetGrid(iris_df, hue="species", height=5)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend();
plt.title("2-D Scatter plot for sepal_length, sepal_width features")
plt.show();


# In[38]:


sns.set_style("whitegrid"); 
sns.pairplot(iris_df, hue="species",vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], height=4);
plt.title("pairplot");
plt.show()


# In[39]:


'sepal_length', 'sepal_width', 'petal_length', 'petal_width'


# In[40]:


#2D Density plot, contors-plot
versicolor = iris_df.loc[iris_df["species"] == "versicolor"]
setosa = iris_df.loc[iris_df["species"] == "setosa"] 
virginica = iris_df.loc[iris_df["species"] == "virginica"]

# For versicolor
sns.jointplot(x="sepal_length", y="sepal_width", data=versicolor, kind="kde");
plt.show();

# For setosa
sns.jointplot(x="sepal_length", y="sepal_width", data=setosa, kind="kde");
plt.show();

# For virginica
sns.jointplot(x="sepal_length", y="sepal_width", data=virginica, kind="kde");
plt.show();

# For versicolor
sns.jointplot(x="petal_length", y="petal_width", data=versicolor, kind="kde");
plt.show();

# For setosa
sns.jointplot(x="petal_length", y="petal_width", data=setosa, kind="kde");
plt.show();

# For virginica
sns.jointplot(x="petal_length", y="petal_width", data=virginica, kind="kde");
plt.show();


# In[ ]:




