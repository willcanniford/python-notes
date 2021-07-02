# Non-negative matrix factorisation (NMF)
Like PCA, NMF is a dimension reduction technique; however, NMF models are interpretable unlike PCA models. 

**All features must be non-negative**. It cannot be applied to all datasets. 

NMF decomposes samples as sums of their parts, and this is what aides the interpretability of the model. For examples, themes in text data and combinations of common patterns in images. 

NMF is available with `sklearn` but the desired number of components must be specified. 

## Word-frequency array example 
Imagine a matrix with rows representing documents or articles, and words as variables with a frequency score. This score is a representation of the appearance of given words in those documents, and may be calculated using 'tf-idf'. 

**tf**: frequency of the word in the document/article. 
**idf**: a weighting that reduces the influence of commons words such as 'the' and 'and'. 

NMF produces components that it learns from the samples and, as with PCA, the dimension of the components matches the dimension of the samples i.e. if there are 4 variables and 2 components you will get 2 component arrays of size 4. 

NMF features are non-negative and are two new features of the dataset. In combination with the components they can be used to approximately reconstruct the original data sample. This is done through multiplying the NMF components by the feature values and then adding them up (a product of matrices). 

```python 
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))
```

When looking at individual rows, you might see that a few have a particular NMF feature that is higher, i.e. feature 4. This means that both rows are predominantly rebuilt using the fourth NMF component, with a component representing a potential topic that the rows have in common. 

As NMF learns interpretable parts, it effectively means that components represent patterns that commonly occur in the samples. If you take a component and look at the variables that are highest in that component, then the groupings of samples may appear more clear. Take the example of work matrices, the words that are highest for any given component might be seen as a topic that those words relate to. 
_This is because the components are at the row level so you can track the component values against the variable names to see the variables with the highest value for each component._

e.g.  
```python
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=variable_names)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3, :]

# Print result of nlargest
print(component.nlargest())
```

For images, the components will represent particular patterns that are seen in the images. 

```python 
from sklearn.decomposition import NMF

model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component) # See working_with_images.md

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)
```

PCA doesn't build interpretable parts, and you can compare the image example using the below code, but note how the resulting images aren't obvious parts of the image. 

```python 
from sklearn.decomposition import PCA

model = PCA(n_components=7)
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
```

Interesting differences can be found here about how NMF and PCA differ in their approach to clustering: 
[Stack Overflow Article](https://stats.stackexchange.com/questions/502072/what-is-the-main-difference-between-pca-and-nmf-and-why-to-choose-one-rather-tha)

PCA: [here](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202)
> Large datasets are increasingly common and are often difficult to interpret. Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance.

This doesn't mention the ability to be able to interpret those values, but talks only about creating new components that try to maintain the highest amount of variance in the data. 

More details about the uses of PCA can be found ['Stack Overflow - PCA for feature selection'](https://stats.stackexchange.com/questions/27300/using-principal-component-analysis-pca-for-feature-selection) and ['What is PCA and how is it used?'](https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186#:~:text=The%20most%20important%20use%20of,variables%2C%20and%20among%20the%20variables.). If the first and second components explain a lot of the variance, then we are able to visual the differences and see potential trends in 2 dimensions. We can also do this through the variable feature space and see which variables are causing the most difference between the samples that we are comparing. 
  

## Building recommender systems
Finding similar samples from a test sample can be done, in theory, as the NMF features of similar samples should also be similar. 

Comparing features can be difficult, as how do you compare the NMF features? The exact feature values might be different due to a shift in the strength of the values. This can manifest in words via the frequency of the works used, and you can have the same selection of words but spread out across the article, rather than a direct level of speech. I assume this would work with grayscale images where you have a lower definition or softer images compared to high contrast values. 

All these versions, when plotted on a scatter plot, lie on the same line through the origin. Therefore, when comparing two samples you can use something called 'cosine similarity' and compare the angles between the two lines. 

__Functionally:__
```python 
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features) # For comparison only?
current_article = norm_features[n, :]
similarities = norm_features.dot(current_article)
```

__Using pandas:__
```python 
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
```

__Full example:__
```python 
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
import pandas as pd 

scaler = MaxAbsScaler()
nmf = NMF(n_components=20)
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
```