import os.path
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp
import torch
from scipy import sparse


def _select_views(raw_views, args, default_indices):
    view_indices = getattr(args, "view_indices", None)
    if view_indices is None:
        view_indices = default_indices
    if len(view_indices) != args.n_views:
        raise ValueError(
            "view_indices length {} does not match n_views {}".format(
                len(view_indices), args.n_views
            )
        )
    if any(index < 0 or index >= len(raw_views) for index in view_indices):
        raise ValueError("view_indices contains an index outside the dataset views")
    return [np.asarray(raw_views[index]).copy() for index in view_indices]


def load_mat(args):
    data_X = []
    label_y = None

    if args.dataset == "LandUse21":
        mat = sio.loadmat(os.path.join(args.data_path, "LandUse_21.mat"))
        raw_views = [
            sparse.csr_matrix(mat["X"][0, 1]).toarray(),
            sparse.csr_matrix(mat["X"][0, 2]).toarray(),
        ]
        data_X = _select_views(raw_views, args, [0, 1])
        label_y = np.squeeze(mat["Y"]).astype("int")

    elif args.dataset == "Caltech101":
        mat = sio.loadmat(
            os.path.join(args.data_path, "2view-caltech101-8677sample.mat")
        )
        X = mat["X"][0]
        data_X = _select_views([X[0].T, X[1].T], args, [0, 1])
        label_y = np.squeeze(mat["gt"]) - 1

    elif args.dataset == 'Animal':
        mat = sio.loadmat(
            os.path.join(args.data_path, 'AWA-7view-10158sample.mat')
        )
        label_y = np.squeeze(mat['gt'])
        data_X = _select_views(
            [mat['X'][0][6].T, mat['X'][0][5].T], args, [0, 1]
        )

    elif args.dataset in ("umpc_food101", "Food101"):
        mat = sio.loadmat(os.path.join(args.data_path, "umpc_food101.mat"))
        img_features = mat["img_features"].astype(np.float32)
        txt_features = mat["txt_features"].astype(np.float32)
        label_y = np.squeeze(mat["labels"])
        data_X = _select_views([img_features, txt_features], args, [0, 1])

    elif args.dataset == "Scene15":
        mat = sio.loadmat(os.path.join(args.data_path, "Scene_15.mat"))
        X = mat["X"][0]
        data_X = _select_views(
            [X[0].astype("float32"), X[1].astype("float32")], args, [1, 0]
        )
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "Reuters":
        mat = sio.loadmat(os.path.join(args.data_path, "Reuters_dim10.mat"))
        raw_views = [
            np.vstack((mat["x_train"][0], mat["x_test"][0])).astype("float32"),
            np.vstack((mat["x_train"][1], mat["x_test"][1])).astype("float32"),
        ]
        data_X = _select_views(raw_views, args, [0, 1])
        label_y = np.squeeze(np.hstack((mat["y_train"], mat["y_test"])))

    elif args.dataset in ("Handwritten", "HandWritten"):
        filename = "Handwritten.mat"
        if not os.path.exists(os.path.join(args.data_path, filename)):
            filename = "HandWritten.mat"
        mat = sio.loadmat(os.path.join(args.data_path, filename))
        raw_views = [np.asarray(view) for view in np.ravel(mat["X"], order="F")]
        data_X = _select_views(raw_views, args, [0, 1])
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "CCV20":
        filename = "CCV20.mat"
        if not os.path.exists(os.path.join(args.data_path, filename)):
            filename = "CCV.mat"
        mat = sio.loadmat(os.path.join(args.data_path, filename))
        raw_views = [np.asarray(view) for view in np.ravel(mat["X"], order="F")]
        data_X = _select_views(raw_views, args, [1, 2])
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "XMediaNet":
        mat = sio.loadmat(os.path.join(args.data_path, "xmedia_deep_2_view.mat"))
        data_X = _select_views([mat['Img'], mat['Txt']], args, [0, 1])
        label_y = np.squeeze(mat['label'])
    else:
        raise KeyError(f"Unknown Dataset {args.dataset}")
    
    label_y = label_y - np.min(label_y)  # make sure label starts from 0

    raw_data = [data.copy() for data in data_X]
    raw_label = label_y.copy()

    # Control the randomness of the data
    rng = np.random.RandomState(1234)
    label_y_view2 = label_y.copy()
    id1 = np.arange(data_X[0].shape[0])
    id2 = id1.copy()
    is_align = np.ones_like(label_y)
    if args.m_ratio > 0:
        for i in range(1, args.n_views):
            inx = np.arange(data_X[i].shape[0])
            rng.shuffle(inx)
            inx = inx[0 : int(args.m_ratio * data_X[i].shape[0])]
            is_align[inx] = 0
            _inx = np.array(inx)
            rng.shuffle(_inx)
            data_X[i][inx] = data_X[i][_inx]
            label_y_view2[inx] = label_y_view2[_inx]
            id2[inx] = id1[_inx]

    if args.data_norm == "standard":
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
    elif args.data_norm == "l2-norm":
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
    elif args.data_norm == "min-max":
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])

    return data_X, label_y, label_y_view2, id1, id2, is_align, raw_data, raw_label


def load_dataset(args):
    data, label1, label2, id1, id2, is_align, raw_data, raw_label = load_mat(args)
    encoder_dim = getattr(args, "encoder_dim", None)
    if encoder_dim:
        expected_dimensions = [int(view_layers[0]) for view_layers in encoder_dim]
        actual_dimensions = [view.shape[1] for view in data]
        if expected_dimensions != actual_dimensions:
            raise ValueError(
                "encoder_dim input dimensions {} do not match loaded views {}".format(
                    expected_dimensions, actual_dimensions
                )
            )
    train_dataset = MultiviewDataset(args.n_views, data, label1, label2, id1, id2, is_align)
    test_dataset = MultiviewDataset(args.n_views, data, label1, label2, id1, id2, is_align)
    return train_dataset, test_dataset


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label1, label2, id1, id2, is_align):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.label1 = label1 - np.min(label1)
        self.label2 = label2 - np.min(label2)
        self.id1 = id1
        self.id2 = id2
        self.is_align = is_align

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype("float32")))
        label1 = torch.tensor(self.label1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        label_align = torch.tensor(self.is_align[idx], dtype=torch.long)
        return idx, data, label1, label2, id1, id2, label_align
