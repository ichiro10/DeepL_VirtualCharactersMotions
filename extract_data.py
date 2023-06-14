import numpy as np

# ['classes', 'clips']
data = np.load('data_styletransfer.npz', allow_pickle=True)

#(Examples, Time frames, joints)
clips   = data['clips']
#(Motion, Styles)
classes = data['classes']

nb_datapoints = classes.shape[0]

# Obtain mean and variance
mean = np.mean(clips, axis=0)
std  = np.std(clips, axis=0)

np.savez('styletransfer_preprocessed.npz', Xmean=mean, Xstd=std)


## Convert to one-hot representation for classification
# ['hiding','showing', 'showingphone', 'stopping', 'waving']
one_hot_motions = np.zeros([nb_datapoints, 5])
one_hot_motions[np.arange(nb_datapoints), classes[:,0]] = 1

# ['Emotionally_stable', 'neutral', 'neurotic']
one_hot_styles  = np.zeros([nb_datapoints, 3])
one_hot_styles[np.arange(nb_datapoints), classes[:,1]] = 1

np.savez('styletransfer_motions_one_hot.npz', one_hot_vectors=one_hot_motions)

np.savez('styletransfer_styles_one_hot.npz', one_hot_vectors=one_hot_styles)