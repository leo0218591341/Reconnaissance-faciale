import time

from io import BytesIO

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import streamlit         as st

from skimage                 import exposure
from sklearn                 import datasets
from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit


@st.cache
def load_data():
    olivetti = datasets.fetch_olivetti_faces()
    return olivetti


def get_figure(data, index):
    figure = plt.figure()
    plt.imshow(x_train[index].reshape(-1, 64), cmap="gray")
    plt.axis('off')
    return figure



############################################################
############################################################


## Sidebar progress bar
st.sidebar.title('DataScience')
st.sidebar.text('Reconaissance faciale')
st.sidebar.header('Pre-processing')

progress         = 0.0
progress_step    = 5.0
progress_bar     = st.sidebar.progress(0.0)
progress_status  = st.sidebar.text(f'{int(progress * 100)} - Initialisation')


## Sidebar settings
st.sidebar.header('Settings')

seed  = st.sidebar.number_input('Random state seed', min_value=0, value=42)


### Title
st.title('DataScience - Reconaissance faciale')


## Dataset
st.header('Olivetti Faces dataset')

# > Progress status
progress_status.text(f'{int(progress * 100)}% - Data loading ...')

# Data load
dataset = load_data()

# > Progress status
progress += 1 / progress_step
progress_bar.progress(progress)

# Description
if st.checkbox('Show description'):
    st.subheader('Description')
    st.text(dataset.DESCR)


## Data split
st.header('Data split between training, testing and validation')

validation_sample_size = st.slider('Validation sample size', 20, 100, 80)
testing_sample_size    = st.slider('Testing sample size', 20, 100, 40)

@st.cache
def split_data(testing_sample_size: int, validation_sample_size: int, rt: int):
    strat_split = StratifiedShuffleSplit(n_splits=10, test_size=testing_sample_size, random_state=rt)
    train_valid_idx, test_idx = next(strat_split.split(dataset.data, dataset.target))
    x_train_valid = dataset.data[train_valid_idx]
    y_train_valid = dataset.target[train_valid_idx]
    x_test = dataset.data[test_idx]
    y_test = dataset.target[test_idx]

    strat_split = StratifiedShuffleSplit(n_splits=10, test_size=validation_sample_size, random_state=rt + 1)
    train_idx, valid_idx = next(strat_split.split(x_train_valid, y_train_valid))
    x_train = x_train_valid[train_idx]
    y_train = y_train_valid[train_idx]
    x_valid = x_train_valid[valid_idx]
    y_valid = y_train_valid[valid_idx]

    return { 'x' : { 'train': x_train, 'test': x_test, 'valid': x_valid },
             'y' : { 'train': y_train, 'test': y_test, 'valid': y_valid },
    }

# > Progress status
progress_status.text(f'{int(progress * 100)}% - Data split ...')

datas = split_data(testing_sample_size, validation_sample_size, seed)
x_train = datas['x']['train']
x_test  = datas['x']['test']
x_valid = datas['x']['valid']
y_train = datas['y']['train']
y_test  = datas['y']['test']
y_valid = datas['y']['valid']

# >Progress status
progress += 1 / progress_step
progress_bar.progress(progress)

if st.checkbox('Show shapes'):
    st.text(f'X train : {x_train.shape} -> Y train : {y_train.shape}')
    st.text(f'X valid : {x_valid.shape} -> Y valid : {y_valid.shape}')
    st.text(f'X test  : {x_test.shape } -> Y test  : {y_test.shape }')


## Sidebar settings
index = st.sidebar.number_input('Image index', min_value=0, value=4, max_value=len(x_train))

# > Progress status
progress_status.text(f'{int(progress * 100)}% - Plotting image ...')

figure = plt.figure()
plt.imshow(x_train[index].reshape(-1, 64), cmap="gray")
plt.axis('off')

st.sidebar.pyplot(figure)

# > Progress status
progress += 1 / progress_step
progress_bar.progress(progress)


## Equalisation
st.header('Histogram equalisation')

limit = st.slider('Clip limit', 0.0, 1.0, 0.01)

# > Progress status
progress_status.text(f'{int(progress * 100)}% - Equalising histograms ...')

if st.checkbox('Equalize x_train', value=True):
    x_train_eq = exposure.equalize_adapthist(x_train, clip_limit=limit)

    left, right = st.columns(2)

    with left:
        buffer = BytesIO()
        figure = plt.figure()

        left.text('Original')

        plt.imshow(x_train[index].reshape(-1, 64), cmap="gray")
        plt.axis('off')

        # Trick for fixed size image
        # figure.savefig(buffer, format="png")
        # st.image(buffer)
        st.pyplot(figure)

    # > Progress status
    progress += 1 / 2 / progress_step
    progress_bar.progress(progress)
    progress_status.text(f'{int(progress * 100)}% - Equalising histograms ...')

    with right:
        buffer = BytesIO()
        figure = plt.figure()

        right.text('Equalised')

        plt.imshow(x_train_eq[index].reshape(-1, 64), cmap="gray")
        plt.axis('off')

        # Trick for fixed size image
        ## figure.savefig(buffer, format="png")
        ## st.image(buffer)
        st.pyplot(figure)

    x_train = x_train_eq

    # > Progress status
    progress += 1 / 2 / progress_step
    progress_bar.progress(progress)
    progress_status.text(f'{int(progress * 100)}% - Equalising histograms ...')


if st.checkbox('Equalize x_test'):
    x_test = exposure.equalize_adapthist(x_test, clip_limit=limit)

if st.checkbox('Equalize x_valid'):
    x_valid = exposure.equalize_adapthist(x_valid, clip_limit=limit)


## PCA Compression
st.header('PCA Compression')

variance = st.slider('Variance', 0.0, 1.0, 0.95)

# > Progress status
progress_status.text(f'{int(progress * 100)}% - PCA Compression ...')

pca = PCA()
pca.fit(x_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= variance) + 1

figure = plt.figure()
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, variance], "k:")
plt.plot([0, d], [variance, variance], "k:")
plt.plot(d, variance, "ko")
plt.annotate("Elbow", xy=(50, 0.80), xytext=(70, 0.65),
             arrowprops=dict(arrowstyle="->"), fontsize=16)
plt.grid(True)
st.pyplot(figure)

# > Progress status
progress += 1 / 2 / progress_step
progress_bar.progress(progress)
progress_status.text(f'{int(progress * 100)}% - PCA Compression ...')

pca = PCA(variance)
pca.fit(x_train)

if st.checkbox('Compress x_train', value=True):
    x_train_pca          = pca.transform(x_train)
    x_train_uncompressed = pca.inverse_transform(x_train_pca)

    left, right = st.columns(2)

    with left:
        buffer = BytesIO()
        figure = plt.figure()

        left.text('Original')

        plt.imshow(x_train[index].reshape(-1, 64), cmap="gray")
        plt.axis('off')

        # Trick for fixed size image
        # figure.savefig(buffer, format="png")
        # st.image(buffer)
        st.pyplot(figure)

    # > Progress status
    progress += 1 / 4 / progress_step
    progress_bar.progress(progress)
    progress_status.text(f'{int(progress * 100)}% - PCA Compression ...')

    with right:
        buffer = BytesIO()
        figure = plt.figure()

        right.text('Compressed')

        plt.imshow(x_train_uncompressed[index].reshape(-1, 64), cmap="gray")
        plt.axis('off')

        # Trick for fixed size image
        ## figure.savefig(buffer, format="png")
        ## st.image(buffer)
        st.pyplot(figure)

    # > Progress status
    progress += 1 / 4 / progress_step
    progress_bar.progress(progress)


if st.checkbox('Compress x_test', value=True):
    x_valid_pca = pca.transform(x_valid)

if st.checkbox('Compress x_valid', value=True):
    x_test_pca = pca.transform(x_test)

# > Progress status
progress = 1.0
progress_bar.progress(1.0)
progress_status.text(f'{int(progress * 100)}% - Done!')


## Unsupervised classification
st.header('Kmeans unsupervised classification')

option = st.selectbox('Which dataset would you like to process?', ('x_train', 'x_train_pca'))

if option == 'x_train':
    data = x_train
if option == 'x_train_pca':
    data = x_train_pca

st.markdown(f'From the elbow, we determined that the probable number of cluster for a variance rate of {variance} would be {d}.')

select_options = [i for i in range(len(x_train_pca))]
lower_bound, higher_bound = st.select_slider('Range' , options=select_options, value=(50, 150))
kmeans_step = st.slider('Step', 1, 10, 5)

def kmeans(data, lower_bound: int, upper_bound: int, step: int, rt: int, bar, bar_status):
    scores = {}
    models = {}

    bar_progress = 0.0

    for k in range(lower_bound, higher_bound + 1, step):
        kmeans = KMeans(n_clusters=k, random_state=rt).fit(data)
        score = silhouette_score(data, kmeans.labels_, metric='euclidean')
        scores[k] = score
        models[k] = kmeans

        bar_progress += 1 / ((higher_bound - lower_bound) / step)
        if bar_progress > 1.0:
            bar_progress = 1.0
        bar.progress(bar_progress)
        bar_status.text(f'{int(bar_progress * 100)}%')

    return scores, models

if st.button('Run Kmeans'):
    bar        = st.progress(0.0)
    bar_status = st.text(f'0%')


    scores, models = kmeans(data, lower_bound, higher_bound, kmeans_step, seed, bar, bar_status)

    scores_dataframe = pd.DataFrame.from_dict(scores, orient='index')
    scores_dataframe.columns = ['score']

    bar.progress(1.0)
    bar_status.text(f'100%')

    x, y = zip(*sorted(scores.items()))

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    best_model = models[best_k]

    buffer = BytesIO()
    figure = plt.figure()

    plt.plot(x, y, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.plot(best_k, best_score, "rs")

    # Trick for fixed size image
    # figure.savefig(buffer, format="png")
    # st.image(buffer)
    st.pyplot(figure)

    st.markdown(f'The best scores is {best_score:.4f} with a number of cluster of {best_k}.')

    with st.expander("Raw scores"):
        st.write(scores_dataframe)

    with st.expander("Clusters"):

        def plot_faces(faces, labels, n_cols=5):
            buffer = BytesIO()

            faces = faces.reshape(-1, 64, 64)
            n_rows = (len(faces) - 1) // n_cols + 1

            figure = plt.figure(figsize=(n_cols, n_rows * 1.1))

            for index, (face, label) in enumerate(zip(faces, labels)):
                plt.subplot(n_rows, n_cols, index + 1)
                plt.imshow(face, cmap="gray")
                plt.axis("off")
                plt.title(label)

            # Trick for fixed size image
            figure.savefig(buffer, format="png")
            st.image(buffer)


        for cluster in np.unique(best_model.labels_):
            st.text(f'Cluster {cluster}')
            in_cluster = best_model.labels_ == cluster
            faces = x_train[in_cluster]
            labels = y_train[in_cluster]
            plot_faces(faces, labels)

# for n, label in enumerate(labels):
    # if label not in clusters:
        # clusters[label] = []
    # clusters[label].append(X_train_valid[n])
    #
# for cluster, images in sorted(clusters.items()):
    # figure  = plt.figure()
    # rows    = 1
    # columns = len(images)
#
    # figure.suptitle(f"Cluster {cluster}")
    # for n, image in enumerate(images):
        # figure.add_subplot(rows, columns, n + 1)
        # plt.imshow(image.reshape(-1, 64), label='_nolegend_', cmap="gray")
        # plt.axis('off')