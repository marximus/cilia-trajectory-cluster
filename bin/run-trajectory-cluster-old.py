"""
Cilia classifier by clustering trajectories to create a codebook of similar motions.

1. Cluster trajectories to create a codebook of similar motions
2. For a new ROI, find which motion it is most similar to in the codebook
3. Label based on most similar.
"""


# Extract descriptors from tracks
desc = np.recarray(shape=(len(tracks),), dtype=list(zip(desc_names, [object] * len(desc_names))))
for desc_name in desc_names:
    for i, X in enumerate(tracks[desc_name]):
        desc[desc_name][i] = SimpleNamespace(X=X)

cv = LeaveOneGroupOut()

for idx, (train, validate) in enumerate(
        cv.split(desc, rois.target.values, groups=rois.patient.values)):
    print('Set {}/{}'.format(idx + 1, cv.get_n_splits(None, None, rois.patient.values)))
    # Remove ROIs that have empty descriptors (because no tracks were computed).
    train = train[~np.in1d(train, no_tracks)]
    validate = validate[~np.in1d(validate, no_tracks)]
    if len(train) == 0:
        quit('Training set is empty after removing ROIs with no tracks')
    if len(validate) == 0:
        print('Validation set is empty after removing ROIs with no tracks. Skipping validation set.')
        continue

    desc_train, y_train = desc[train], rois.target.values[train]
    desc_val, y_val = desc[validate], rois.target.values[validate]

# Generate codebooks
print('Creating codebooks (n_iter={}, n_desc={}, n_words={})'.format(n_iter, n_desc, n_words))
codebooks = dict()
for desc_name in desc_names:
    # concatenate descriptors from all ROIs into a n_descriptors x n_features array
    X_desc = np.vstack(desc_train[desc_name][i].X for i in range(len(desc_train)))
    y_desc = np.array([y for y, d in zip(y_train, desc_train[desc_name]) for _ in range(len(d.X))])
    if n_desc:
        rand = np.random.choice(len(X_desc), size=np.amin([n_desc, len(X_desc)]), replace=False)
        X_desc, y_desc = X_desc[rand, :], y_desc[rand]
    cb, label, inertia = generate_codebook(X_desc, y_desc, n_iter=n_iter, n_words=n_words)
    codebooks[desc_name] = cb

# quantization - assign each descriptor to the most representative visual word
for desc_name in desc_train.dtype.names:
    codebook = codebooks[desc_name]
    for i in range(len(desc_train[desc_name])):
        codes, dist = vq.vq(desc_train[desc_name][i].X, codebook)
        hist = np.bincount(codes, minlength=len(codebook))
        hist = hist / hist.sum()
        desc_train[desc_name][i].codes = codes
        desc_train[desc_name][i].dist = dist
        desc_train[desc_name][i].hist = hist
for desc_name in desc_val.dtype.names:
    codebook = codebooks[desc_name]
    for i in range(len(desc_val[desc_name])):
        codes, dist = vq.vq(desc_val[desc_name][i].X, codebook)
        hist = np.bincount(codes, minlength=len(codebook))
        hist = hist / hist.sum()
        desc_val[desc_name][i].codes = codes
        desc_val[desc_name][i].dist = dist
        desc_val[desc_name][i].hist = hist

# Create kernel for each descriptor
print('Creating kernel (gamma={})'.format(gamma))
# train
kernels_train = []
for desc_name in desc_names:
    hist_train = np.vstack(desc_train[desc_name][i].hist for i in range(len(desc_train)))
    chi2_train = pairwise.chi2_kernel(hist_train, gamma=gamma)
    kernels_train.append(chi2_train)
# validation
kernels_val = []
for desc_name in desc_names:
    hist_train = np.vstack(desc_train[desc_name][i].hist for i in range(len(desc_train)))
    hist_val = np.vstack(desc_val[desc_name][i].hist for i in range(len(desc_val)))
    chi2_val = pairwise.chi2_kernel(hist_val, hist_train, gamma=gamma)
    kernels_val.append(chi2_val)

# Combine kernels to create a single kernel for training and testing data
K_train = np.zeros(shape=kernels_train[0].shape)
for kernel, weight in zip(kernels_train, weights):
    # mu = 1.0 / kernel.mean()
    # K_train += weight * np.exp(-mu * kernel)
    K_train += weight * kernel
K_val = np.zeros(shape=kernels_val[0].shape)
for kernel, weight in zip(kernels_val, weights):
    # mu = 1.0 / kernel.mean()
    # K_val += weight * np.exp(-mu * kernel)
    K_val += weight * kernel

skf = StratifiedShuffleSplit(n_splits=n_val_preds, train_size=0.9)
for ind, _ in skf.split(K_train, y_train):
    K_train_nested, y_train_nested = K_train[ind][:, ind], y_train[ind]
    K_val_nested = K_val[:, ind]

    clf = svm.NuSVC(nu=nu, kernel='precomputed')
    clf.fit(K_train_nested, y_train_nested)
    y_pred = clf.predict(K_val_nested)
    for roi, pred in zip(validate, y_pred):
        y_preds[roi].append(pred)

# Use a majority voting scheme to get ROI predictions. Classify any ROIs with no trajectories
# as being abnormal.
y_preds = [majority_vote(preds) for preds in y_preds]
for i in no_tracks:
    y_preds[i] = 1

df = pd.DataFrame(data={'y_pred': y_preds, 'y_true': rois.target.values, 'patient': rois.patient.values})
patients = {'name': [], 'y_pred': [], 'y_true': [], 'n_rois': []}
for patient, group in df.groupby('patient'):
    # get patient target from roi targets
    if not np.all(group['y_true'].values[0] == group['y_true'].values):
        print('All ROIs of a patient must have same y_true')
    patients['name'].append(patient)
    patients['y_true'].append(group['y_true'].values[0])
    patients['y_pred'].append(majority_vote(group['y_pred']))
    patients['n_rois'].append(len(group))
patients = pd.DataFrame(data=patients, columns=['name', 'n_rois', 'y_true', 'y_pred'])

# Set the y_preds which are -1 to opposite of true value so that it doesn't mess the
# classification_accuracy function up below
for i in np.flatnonzero(patients['y_pred'] == -1):
    true = patients.ix[i, 'y_true']
    patients.ix[i, 'y_pred'] = 1 if true == 0 else 0

print_full(patients)
print(classification_report(patients['y_true'].values, patients['y_pred'].values))


##########################################################
# Merge trajectory clusters found in ROIs.
##########################################################
metric = 'euclidean_sum'    # distance between trajectory descriptors
method = 'single'           # distance between clusters
max_d = None                # cutoff for dendrogram - user enters on command line

# remove trajectories that do not belong to any clusters
mask = (track_clusters != -1)
tracks = tracks[mask]
track_clusters = track_clusters[mask]

# unique cluster ids and indices of trajectory descriptors in each cluster
cluster_ids, unq_inv, unq_cnt = np.unique(track_clusters, return_counts=True, return_inverse=True)
cluster_indices = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
n_clusters = len(cluster_ids)
print('n_clusters = {}'.format(n_clusters))

# cluster_ids, cluster_indices = tcluster.unique_clusters(clusters)
# n_clusters = len(cluster_ids)

# inter-cluster distances
print('trajectory: {}'.format(tracks['trajectory'].shape))
print('clusters: {}'.format(track_clusters.shape))
C = tdistance.cluster_pdist(tracks['trajectory'], cluster_indices, method)
print(C.shape)

# hierarchical agglomerative clustering of clusters
Z = hierarchy.linkage(C, method=method)

# save clustermap
# print('Saving clustermap ...')
# clustergrid = sns.clustermap(squareform(C), row_linkage=Z, col_linkage=Z, figsize=(8*4, 6*4))
# clustergrid.savefig(os.path.join(BASE_OUTPUT_DIR, 'clustermap.pdf'))

# set leaf colors for dendrogram to green for normal and red for abnormal
leaf_colors = dict()
for c_id, c_indices in zip(cluster_ids, cluster_indices):
    targets = tracks['target'][c_indices]
    t = targets[0]
    if not np.all(targets == t):
        quit('must all have same targets')
    leaf_colors[c_id] = 'green' if t == 0 else 'red' if t == 1 else 'black'

# get cut distance from user if not specified
if max_d is None:
    plot.dendrogram(Z, figsize=(32, 24), save_path=os.path.join(BASE_OUTPUT_DIR, 'dendrogram.pdf'))
    max_d = float(input('Cutoff distance: '))

# save image showing the clusters generated by cutting the tree at the specified distance
# plot.dendrogram(os.path.join(BASE_OUTPUT_DIR, 'dendogram_cut.pdf'), Z, max_d)
fname = os.path.join(BASE_OUTPUT_DIR, 'dendogram_cut.pdf')
print('Saving {}'.format(fname))
plot.dendrogram(Z, max_d, figsize=(32, 24), save_path=fname)

# cut the tree at the specified distance to create multiple subtrees (clusters)
# L - contains the node id of the root node for each subtree (cluster)
# M - contains the cluster ids
# For example: if L[3]=2 and M[3]=8, the flat cluster with id 8 has as it's leader linkage node 2
# rd - maps node ids to ClusterNode references
T = hierarchy.fcluster(Z, max_d, criterion='distance')
L, M = hierarchy.leaders(Z, T)
_, rd = hierarchy.to_tree(Z, rd=True)

# output directory for visualizations of merged clusters
outdir = os.path.join(BASE_OUTPUT_DIR, 'merged_clusters')
if not os.path.exists(outdir):
    os.mkdir(outdir)

# 1. open a terminal and run `ssh -Y vivaldi`
# 2. in open terminal, run `echo $DISPLAY`
# 3. value returned from `echo $DISPLAY` should be used as environment variable
os.environ['DISPLAY'] = ':10.0'

# visualize each merged cluster
print('Saving merged clusters ...')
for node_id, cluster_id in tqdm.tqdm(zip(L, M), total=len(L)):
    # subtree where leaf nodes are the original cluster ids that were merged
    subtree = rd[node_id]
    if subtree.get_count() <= 1:
        continue

    orig_cluster_ids = subtree.pre_order()

    # map original cluster id to trajectories for each original cluster in
    # merged cluster
    tr_dict = {c_id: tracks[track_clusters == c_id] for c_id in orig_cluster_ids}

    # output file name contains ids of merged clusters. The file name cannot
    # be more than 255 characters (including extension).
    fname = "_".join(map(str, orig_cluster_ids))[:251] + ".mp4"
    tree.vis_cluster_tree(os.path.join(outdir, fname), subtree, tr_dict, fps=15)
