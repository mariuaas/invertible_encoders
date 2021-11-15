from .. import (
    os, pickle, gzip, np, Image, torch, torchvision, T, json
)

# TODO: Add docstrings!
# TODO: Possible cleanup, fix some mismatch in attribute names for CIFAR and MNIST
# TODO: Might be better to move default root folder one step further down.

def get_emnist_dict(subdict: str = None, root: str = '../../data') -> dict:
    with open(f'{root}/emnist.json', "r") as json_file:
        emnist_dict = json.load(json_file)

    if subdict is None or subdict not in emnist_dict:
        return emnist_dict

    else:
        return emnist_dict[subdict]


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    Based on source:
    https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class SimpleImageDataset:

    def __init__(self, root, transform=None, filelist=None):
        if filelist is None:
            self.filelist = os.listdir(root)
        else:
            self.filelist = filelist

        self.root = root
        if transform is not None:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.ToTensor()

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, key):
        return self.transform(
            self.pil_loader(
                os.path.join(self.root, self.filelist[key])
            )
        )


def get_coco(
        train=True,
        root='/itf-fi-ml/shared/IN5400/dataforall/mandatory2/data/coco',
        filter_large=True,
        transform=None
    ):
        prefix = 'train' if train else 'val'
        ext = f'{prefix}2017'
        listfile = f'../../data/coco_largefiles_{prefix}.pickle'
        with open(listfile, 'rb') as infile:
            filelist = pickle.load(infile)
        return SimpleImageDataset(
            os.path.join(root, ext),
            transform,
            filelist,
        )


class EMNIST:

    valid_splits = ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']

    def __init__(self, split, root='../../data/', train=True, transform=None):
        assert split in self.valid_splits, f'Split {split} is not a valid EMNIST dataset.'

        self.train = train
        self.root = root
        self.base_folder = 'EMNIST/raw/'
        self.size = (28, 28)
        self.filenames = {
            'train': {
                'data': f'emnist-{split}-train-images-idx3-ubyte.gz',
                'targets': f'emnist-{split}-train-labels-idx1-ubyte.gz'
            },
            'test': {
                'data': f'emnist-{split}-test-images-idx3-ubyte.gz',
                'targets': f'emnist-{split}-test-labels-idx1-ubyte.gz'
            },
        }

        self.fliprot = True
        self.get_classes(split)
        self.load_binary_data()
        self.transform = transform

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def get_classes(self, split):
        with open(f'{self.root + self.base_folder}emnist-{split}-mapping.txt') as infile:
            self.classes = []
            for line in infile:
                i, c = line.split()
                self.classes.append(chr(int(c)))

    def load_binary_data(self):
        self.data = np.empty((0,28,28)).astype(np.uint8)
        self.targets = np.empty((0,1)).astype(int)

        path = self.root + self.base_folder
        mode = 'train' if self.train else 'test'

        with gzip.open(path + self.filenames[mode]['data'], 'rb') as infile:
            img_train = np.frombuffer(infile.read(), np.uint8, offset=16).reshape(-1, 28, 28)

        with gzip.open(path + self.filenames[mode]['targets'], 'rb') as infile:
            lab_train = np.frombuffer(infile.read(), np.uint8, offset=8).reshape(-1, 1)

        self.data = np.vstack([self.data, img_train])
        self.targets = np.vstack([self.targets, lab_train])

        self.targets = self.targets[:,0]

        if self.fliprot:
            self.data = np.fliplr(np.rot90(self.data, 1, axes=(1,2)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.transform(Image.fromarray(self.data[index], mode='L')),
            self.targets[index]
        )


def get_mnist(split, root='../../data/', train=True, transform=None):
    if split in EMNIST.valid_splits:
        return EMNIST(split, root=root, train=train, transform=transform)
    else:
        raise NotImplementedError(f'Split type {split} not implemented.')


def get_cifar(n=10, download=False, root='../../data/', train=True, transform=None, return_indices=False):
    if transform is None:
        transform = []

    transform = T.Compose(transform + [T.ToTensor()])
    if n == 10:
        dataclass = torchvision.datasets.CIFAR10

    elif n == 100:
        dataclass = torchvision.datasets.CIFAR100

    else:
        raise ValueError('CIFAR only available in 10 or 100 classes mode.')

    if return_indices:
        dataclass = dataset_with_indices(dataclass)

    data = dataclass(
        root=root,
        train=train,
        download=download,
        transform=transform
    )

    setattr(data, 'size', (32, 32))
    return data
