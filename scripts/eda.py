#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">☁️ - Cloudy regions segmentation 👨‍💻🔬</h1>
# 
# <h2 align="center">EDA</h2>
# <p style="text-align:center">
#    Thomas Bury, Afonso Alves, Daniel Staudegger<br>
#    Allianz<br>
# </p>
# 
# <b style="color:darkgold"> EDA is based on this [Kaggle Kernel](https://www.kaggle.com/ekhtiar/eda-find-me-in-the-clouds/notebook)</b>
# 

# In[1]:


import numpy as np 
import pandas as pd
import os
import cv2
import yaml
import seaborn as sns
import scicomap as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import patches as patches
from pprint import pprint

# Custom package for the project, save all the functions into appropriate sub-packages
from pyreidolia.plot import set_my_plt_style, plot_cloud, plot_rnd_cloud, draw_label_only
from pyreidolia.mask import bounding_box, rle_to_mask, get_binary_mask_sum


# # Setting global matplotlib style
# If you don't like, just revert back to defaults using `plt.style.use('default')`

# In[2]:


# A nicer style for mpl
set_my_plt_style(height=6, width=8, linewidth=1.5)

# A better colormap


sc_map = sc.ScicoSequential(cmap='tropical')
sc_map.unif_sym_cmap(lift=None, 
                     bitonic=False, 
                     diffuse=True)
sc_cmap = sc_map.get_mpl_color_map()

mpl.cm.register_cmap("tropical", sc_cmap)


# # Load the config file for the paths
# To avoid to hardcode the paths in a versioned file, let's create a `paths.yml` which will **not** be versioned. So that the paths are not overwritten when we pull or merge from the GitHub repo. The `paths.yml` should have a structure like:
# 
# ```yml
# # data
# data:
#   test: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/data/test_images"
#   train: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/data/train_images"
#   docs: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/data/"
# 
# # Path to store all notebooks, ideally not versioned
# notebooks: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/notebooks/"
# 
# # Path to store all outputs (correlations, jsons, excel, etc)
# output: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/output/"
# 
# # Path to store all python scripts, for versioning
# scripts: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/scripts/"
# 
# # Path to studies
# studies: "C:/Users/xtbury/Documents/Projects/segmentation_cloudy_regions/studies/"
# ```

# In[3]:


def string_print(df):
    return print(df.to_string().replace('\n', '\n\t'))


# In[4]:


# Where is my yaml ? "C:/Users/xtbury/Documents/Projects/Pyreidolia/paths.yml"

paths_yml = input("where is the paths.yml config file?")
with open(paths_yml, "r") as ymlfile:
    path_dic = yaml.load(ymlfile, Loader=yaml.FullLoader)

pprint(path_dic)


# In[5]:


train_csv_path = path_dic['data']['docs'] + 'train.csv'
train_data = path_dic['data']['train'] 
test_data = path_dic['data']['test'] 


# # Load the data doc

# In[6]:


train_csv_path


# In[7]:


train_doc = pd.read_csv(train_csv_path)
print(train_doc.head())


# In[8]:


# mix of naming convention, let's fix it
train_doc = train_doc.rename({'Image_Label': 'image_label', 'EncodedPixels': 'encoded_pixels'}, axis=1)


# # Prettify documentation

# Split the image labels to get the type and the ID.

# In[9]:


# image id and class id are two seperate entities and it makes it easier to split them up in two columns
# train_doc[] = train_doc['Image_Label'].str.split('_')[0] 
# train_doc['Label'] = train_doc['Image_Label'].str.split('_')[1] 
train_doc[['image_id', 'label']] = train_doc['image_label'].str.split('_', 1, expand=True)
# lets create a dict with class id and encoded pixels and group all the defaults per image
train_doc['label_encodedpix'] = list(zip(train_doc['label'].values, train_doc['encoded_pixels'].values)) 
# Let's create a boolean variable if there is a mask attached to a label
train_doc['is_mask']= ~train_doc.encoded_pixels.isnull().values

train_doc.head()


# Check the unique values of `label`

# In[10]:


print(train_doc.label.unique())


# Basic information, as the number of non-null (non NaNs) and data type

# # Are all the images labelled?

# In[11]:


print(train_doc.info())


# In[12]:


n_nans = train_doc.isnull().sum().loc["encoded_pixels"]
print(train_doc.isnull().sum())


# In[13]:


pprint(f"The percentage of instances without encoded_pixels: {100*n_nans/len(train_doc):.2f}%")


# ## How many unique values?

# In[14]:


train_doc.nunique()


# # Labels distribution
# 
#  * Overall, non-grouped, labels distribution: select rows with a bounding box
#  * Per Image labels distribution: Select instances (rows) with a bouding box, group by image ID (a image might have more than one bouding box). Then count the labels per image ID and list which are those labels.

# In[15]:


fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.countplot(y='label',
                   data=train_doc.loc[train_doc.is_mask, :],
                   order=train_doc.loc[train_doc.is_mask, 'label'].value_counts().index,
                   palette='tropical');

ax.set_title('Distribution of labels')
ax.set_ylabel('Label');


# Slightly non-uniform distribution, that might bias the model
# 
# ## Pattern distribution

# In[16]:


# lets group each of the types and their mask in a list so we can do more aggregated counts
# Select the rows with a bounding box using train_doc.is_bb
# group by image ID, a image might have more than 1 BB
# count the labels per image and list the attached labels
grouped_labels = train_doc.loc[train_doc.is_mask, :].groupby('image_id')[['is_mask', 'label']].agg({'is_mask': 'sum', 'label': lambda x: list(x)})
grouped_labels["label_comb"] = grouped_labels["label"].str.join("-")
print(grouped_labels.head())


# In[17]:


fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.countplot(data=grouped_labels, y="is_mask", palette='tropical', ax=ax)
ax.set_title('Distribution of number of labels per image')
ax.set_ylabel('Number of labels');


# 4 types of cloud formation in one image is very rare. Only one type of cloud formation in the image is common.

# In[18]:


fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.countplot(y='label_comb',
                   data=grouped_labels,
                   order=grouped_labels['label_comb'].value_counts().index,
                   palette='tropical')

ax.set_title('Distribution of labels per image')
ax.set_ylabel('Label(s)');


# All combination of cloud formations appearing together is a possibility, and the combinations between Sugar, Fish, and Gravel are more likely than with Flower cloud formation. We note that sugar appears in 7 of the 8 most frequent patterns.

# # Explore Images, masks and their bounding boxes

# In[19]:


colors = [(0,0,255), (255,0,0), (0,255,0), (255,255,0)]
image_name = '7405a00.jpg'

rles = train_doc[train_doc['image_id']==image_name]['encoded_pixels'].reset_index(drop=True)
image_start = plt.imread(os.path.join(train_data, image_name))

fig, ax = plt.subplots()
ax.imshow(image_start) 
plt.show()


# In[20]:


img = image_start = plt.imread(os.path.join(train_data, train_doc['image_id'][0]))
mask_decoded = rle_to_mask(train_doc['label_encodedpix'][0][1], size=img.shape)
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
ax[0].imshow(img)
ax[1].imshow(mask_decoded);


# As explained in the paper, a mask not outlining the exact clouds but roughly the area with the same kind of patterns.
# 
# ## Difference between mask and bouding box

# In[21]:


grouped_masks = train_doc.loc[train_doc.is_mask, :].groupby('image_id')['label_encodedpix'].apply(list)


# In[22]:


_ = plot_rnd_cloud(img_path=train_data, grouped_masks=grouped_masks, n_samples=9, figsize=(20,20))


# ## Check cloud patterns
# 
# what do they look like?

# In[23]:


for label in train_doc.label.unique():
    draw_label_only(train_df=train_doc, train_path=train_data, label=label)


# # Patterns surface distribution
# 
# What is the typical surface of a given cloud formation? Are some of the patterns less extented than others?

# In[25]:


get_ipython().run_cell_magic('time', '', "# should find a faster method\ntrain_doc['mask_pixel_sum'] = np.nan\ntrain_doc.loc[train_doc.is_mask, 'mask_pixel_sum'] = train_doc.loc[train_doc.is_mask, :].apply(lambda x: get_binary_mask_sum(x['encoded_pixels']), axis=1)")


# In[30]:


sns.displot(train_doc, x="mask_pixel_sum", hue="label", stat="density", element="step", alpha=.4);


# In[34]:


sns.displot(train_doc, x="mask_pixel_sum", hue="label", kind="ecdf");


# It seems that the surface distributions are more or less spanning the same range. Sugar seems to be less extended (higher proba to span smaller area). Is this genuine or due to human labelling.
