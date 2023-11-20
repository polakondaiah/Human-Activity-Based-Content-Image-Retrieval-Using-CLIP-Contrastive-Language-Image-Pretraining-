# Human-Activity-Based-Content-Image-Retrieval-Using-CLIP-Contrastive-Language-Image-Pretraining-

Project Description: Human Activity-Based Content Retrieval CLIP-Contrastive-Language-Image-Pretraining

Problem Overview
Image-to-image retrieval,particularly within the context of human activity recognition. The primary objective is to implement a system capable of accurately ranking a collection of images based on their resemblance to a given query image.

Dataset Overview

The dataset is a rich collection of images portraying a variety of human activities such as
"hugging," "bicycling," and "eating." The dataset is organized as follows:

Query Data (query_images/): Comprises 150 images with mixed labels, designated for querying
the system.

Gallery Data (gallery/): A set of 1,000 images with mixed labels, serving as the retrieval gallery.

JSON Mapping Files: three JSON files (query_info.json, gallery_info.json) provide image-to-label mappings for training and testing, respectively.

Each image is categorized under one of 15 human activity classes: 'calling', ’clapping’, ’cycling’,
’dancing’, ‘drinking’, ‘eating’, ‘fighting’, ‘hugging’, ‘laughing’, ‘listening_to_music’, ‘running’,
‘sitting’, ‘sleeping’, 'texting', and ‘using_laptop’.

Project Objectives

a. Utilize the 150 query images to evaluate the model's performance, comparing against the
gallery of 1,000 images.
b. Employ Mean Average Precision (mAP) at K={1, 10, 50} and mean rank as evaluation
metrics.
