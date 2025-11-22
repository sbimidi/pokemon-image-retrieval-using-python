#Name- sandeep reddy bimidi
#ASU ID - 1222081185
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

mat = scipy.io.loadmat('/Users/sandeepreddy/Downloads/HW8-FinalProject 2/DeepNetFeature/feature_vgg_f.mat')

# features are extrated in other code just to gather the dict words
features = mat['image_feat']['feat'][0, :]
image_names = mat['image_feat']['name'][0, :]

features = np.array([np.squeeze(np.asarray(f)) for f in features])

pca = PCA(n_components=2)
transformed_features = pca.fit_transform(features)

fig, ax = plt.subplots(figsize=(15, 10))

#  PCA coordinates are used here.
for i, (x, y) in enumerate(transformed_features):
    img_path = f'/Users/sandeepreddy/Downloads/HW8-FinalProject 2/Data/Database/{image_names[i][0]}'
    img = Image.open(img_path)
    
    # image box  for the image
    imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom as necessary
    
    #  annotation box is used for plotting the images
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    
    ax.add_artist(ab)

#axis labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA Feature Embedding with Images')

ax.set_xlim([transformed_features[:, 0].min() - 1, transformed_features[:, 0].max() + 1])
ax.set_ylim([transformed_features[:, 1].min() - 1, transformed_features[:, 1].max() + 1])

plt.show()
