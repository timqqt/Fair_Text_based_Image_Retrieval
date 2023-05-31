# Mitigating Test-time Bias for Fair Image Retrieval

We address the challenge of generating fair and unbiased image retrieval results given neutral textual queries (with no explicit gender or race connotations), while maintaining the utility (performance) of the underlying vision-language (VL) model. Previous methods aim to disentangle learned representations of images and text queries from gender and racial characteristics. However, we show these are inadequate at alleviating bias for the desired equal representation result, as there usually exists test-time bias in the target retrieval set. So motivated, we introduce a straightforward technique, Post-hoc Bias Mitigation (PBM), that post-processes the outputs from the pre-trained vision-language model. We evaluate our algorithm on real-world image search datasets, Occupation 1 and 2, as well as two large-scale image-text datasets, MS-COCO and Flickr30k. Our approach achieves the lowest bias, compared with various existing bias-mitigation methods, in text-based image retrieval result while maintaining satisfactory retrieval performance.

## Data

Occupation 1 can be acquired [here](https://github.com/mjskay/gender-in-image-search) and Occupation 2 can be downloaded from  [here](https://drive.google.com/drive/folders/1j9I5ESc-7NRCZ-zSD0C6LHjeNp42RjkJ?usp=drive_link). MS-COCO is available from [here](https://cocodataset.org/#home). Flickr30k dataset can be access via [here](https://shannon.cs.illinois.edu/DenotationGraph/.)

## Environment setup

~~~
conda create --name fairIR
conda activate fairIR
pip install -r requirements.txt
~~~


## Running the code
- reproduce our results using Debias_notebook.ipynb


