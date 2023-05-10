import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)


class AffectnetPairExpression(Dataset):
    def __init__(self, data_root, path_to_csv, n_expression=8, image_shape=(3, 256, 256), augment=False, dataset_len=-1,
                 verbose=0):
        super(AffectnetPairExpression, self).__init__()

        # root folder containing all the images
        self.data_root = data_root

        # get dataset
        data_df = pd.read_csv(path_to_csv)

        # Extract only the relevant columns
        data_df = data_df[['subDirectory_filePath', 'expression', 'valence', 'arousal']]
        # data_df = data_df.sample(250, random_state=5, ignore_index=True)

        # Remove expression label 8 (this corresponds to None expression)
        self.data_df = data_df[data_df.expression != 8]

        # reset index
        self.data_df.reset_index(inplace=True)

        # get only one sample per class
        # self.data_df.groupby('expression').first().reset_index()
        # self.data_df = pd.DataFrame(self.data_df.groupby('expression').head(2).reset_index(drop=True))

        # Number of expressions to consider, mostly 7 or 8
        self.n_expression = n_expression
        # input image size to the neural network
        self.image_shape = image_shape

        # Augment
        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).

            self.transform = transforms.Compose([
                transforms.Resize((self.image_shape[1] + 32, self.image_shape[2] + 32)),
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop((self.image_shape[1], self.image_shape[2])),
                transforms.ToTensor()
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.Resize((self.image_shape[1] + 32, self.image_shape[2] + 32)),
                transforms.CenterCrop((self.image_shape[1], self.image_shape[2])),
                transforms.ToTensor()
            ])

        self.dataset_len = dataset_len

        if verbose:
            print(f'**** Dataframe ****\n {self.data_df}\n')

        self.group_examples()

    def group_examples(self):
        temp_df = self.data_df.copy()

        self.group_dict = {}

        # 0, 1, 2, 3, .., 6 or 7
        expressions_list = range(0, self.n_expression)
        for i in expressions_list:
            self.group_dict[i] = list(temp_df[temp_df.expression == i].index)

    def __len__(self):
        if self.dataset_len < 0:
            # if sf.dataset_len is -5, then 5 * self.data_df.shape[0]
            return abs(self.dataset_len) * self.data_df.shape[0]
        else:
            return self.dataset_len

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class,
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn
            the similarity between two different images representing the same class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """
        # pick some random class for the first image
        selected_class = random.randint(0, self.n_expression-1)

        # pick a random index for the first image in the grouped indices based on the label of the class
        random_index_1 = random.choice(self.group_dict[selected_class])

        # get the first image
        image_1_path = self.data_df.iloc[random_index_1].subDirectory_filePath

        # same class
        if index % 2 == 0:

            # pick a random index for the second image
            random_index_2 = random.choice(self.group_dict[selected_class])

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.choice(self.group_dict[selected_class])

            # get the second image
            image_2_path = self.data_df.iloc[random_index_2].subDirectory_filePath

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.long)

        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, self.n_expression-1)

            # ensure that the class of the second image is not same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, self.n_expression-1)

            # pick a random index for the second image
            random_index_2 = random.choice(self.group_dict[other_selected_class])

            # get the second image
            image_2_path = self.data_df.iloc[random_index_2].subDirectory_filePath

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.long)

        # Load both the images
        image1 = Image.open(os.path.join(self.data_root, image_1_path)).convert("RGB")
        image2 = Image.open(os.path.join(self.data_root, image_2_path)).convert("RGB")

        # Transform
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # get other labels
        valence = [self.data_df.iloc[random_index_1].valence, self.data_df.iloc[random_index_2].valence]

        arousal = [self.data_df.iloc[random_index_1].arousal, self.data_df.iloc[random_index_2].arousal]

        categories = [self.data_df.iloc[random_index_1].expression, self.data_df.iloc[random_index_2].expression]

        random_index = [random_index_1, random_index_2]

        return dict(image1=image1, image2=image2, target=target, valence=torch.FloatTensor(valence),
                    arousal=torch.FloatTensor(arousal), categories=torch.FloatTensor(categories),
                    random_index=torch.FloatTensor(random_index),
                    delta_valence=torch.tensor(valence[0] - valence[1], dtype=torch.float),
                    delta_arousal=torch.tensor(arousal[0] - arousal[1], dtype=torch.float),
                    image_1_path=image_1_path, image_2_path=image_2_path)


# Data loader for getting an image, as opposed to *pair* of images.
class AffectnetIdenticalPair(Dataset):
    """This class is used to get same images as a pair. This is used when getting features from the trained model
    and retraining features using MTCLAR + SL to predict categorical classes or valence and arousal."""

    def __init__(self, data_root, path_to_csv, n_expression=8, image_shape=(3, 256, 256), augment=False, verbose=0):
        super(AffectnetIdenticalPair, self).__init__()

        # root folder containing all the images
        self.data_root = data_root

        # get dataset
        data_df = pd.read_csv(path_to_csv)

        # Extract only the relevant columns
        data_df = data_df[['subDirectory_filePath', 'expression', 'valence', 'arousal']]
        # self.data_df = self.data_df.head(2000)

        # Remove expression label 8 (this corresponds to None expression)
        self.data_df = data_df[data_df.expression != 8]

        # reset index
        self.data_df.reset_index(inplace=True)

        # get only one sample per calss
        # self.data_df.groupby('expression').first().reset_index()
        # self.data_df = pd.DataFrame(self.data_df.groupby('expression').head(2).reset_index(drop=True))

        # Number of expressions to consider, mostly 7 or 8
        self.n_expression = n_expression
        # input image size to the neural network
        self.image_shape = image_shape

        # Augment
        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).

            self.transform = transforms.Compose([
                transforms.Resize((self.image_shape[1] + 32, self.image_shape[2] + 32)),
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop((self.image_shape[1], self.image_shape[2])),
                transforms.ToTensor()
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.Resize((self.image_shape[1] + 32, self.image_shape[2] + 32)),
                transforms.CenterCrop((self.image_shape[1], self.image_shape[2])),
                transforms.ToTensor()
            ])

        if verbose:
            print(f'**** Dataframe ****\n {self.data_df}\n')

    def __len__(self):
        return 1 * self.data_df.shape[0]

    def __getitem__(self, index):
        # get the first image
        image_1_path = self.data_df.iloc[index].subDirectory_filePath

        # Load both the images
        image1 = Image.open(os.path.join(self.data_root, image_1_path)).convert("RGB")

        # Transform
        if self.transform:
            image1 = self.transform(image1)

        # get other labels
        valence = [self.data_df.iloc[index].valence, self.data_df.iloc[index].valence]

        arousal = [self.data_df.iloc[index].arousal, self.data_df.iloc[index].arousal]

        categories = [self.data_df.iloc[index].expression, self.data_df.iloc[index].expression]

        random_index = [index, index]

        return dict(image1=image1, image2=image1, target=torch.FloatTensor(1), valence=torch.FloatTensor(valence),
                    arousal=torch.FloatTensor(arousal), categories=torch.FloatTensor(categories),
                    random_index=torch.FloatTensor(random_index),
                    delta_valence=torch.tensor(valence[0] - valence[1], dtype=torch.float),
                    delta_arousal=torch.tensor(arousal[0] - arousal[1], dtype=torch.float),
                    image_1_path=image_1_path, image_2_path=image_1_path)


# poor man's dataset
class AffectnetFeature(Dataset):
    """
        Characterizes a dataset for PyTorch
    """
    def __init__(self, Feats, categories, valence, arousal):
        """Initialization"""
        self.Feats = Feats
        self.categories = categories
        self.valence = valence
        self.arousal = arousal

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.categories)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        return dict(feats=self.Feats[idx, :], categories=int(self.categories[idx, ]),
                    valence=self.valence[idx, ], arousal=self.arousal[idx, ])


def visualise_AffectnetPair(dataset, batch_size=1):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_dict = next(iter(dataloader))

    # plot data
    # To have dark background
    # plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Visualise first sample in the batch
    axes[0].imshow(data_dict["image1"][0, :, :, :].permute(1, 2, 0))
    axes[0].set_title(f'Valence: {data_dict["valence"][0, 0]:.2f}\nArousal: {data_dict["arousal"][0, 0]:.2f}' +
                      f'\nExpression: {data_dict["categories"][0,0]}')
    axes[1].imshow(data_dict["image2"][0, :, :, :].permute(1, 2, 0))
    axes[1].set_title(f'Valence: {data_dict["valence"][0, 1]:.2f}\nArousal: {data_dict["arousal"][0, 1]:.2f}' +
                      f'\nExpression: {data_dict["categories"][0,1]}')
    fig.suptitle(f'Similarity Label: {int(data_dict["target"][0])}')
    plt.show()
    return


def total_pairs_AffectnetPair(dataset, batch_size=32):
    """
    Calculate total pairs
    :param dataset: Torch dataset instance
    :param batch_size: batch size
    :return: all pairs y
    """
    # load data to PyTorch dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    y = np.zeros((len(dataloader), batch_size))
    for i, data_dict in tqdm(enumerate(dataloader)):
        y[i] = data_dict['target'].numpy()
        # y.append(data_dict['target'].numpy())

    print(y)
    print(10*'*')
    print("It's shape:", y.shape)
    return y


if __name__ == "__main__":
    data_root = "../../data/AffectNet/raw_images"
    path_to_csv = "../../data/AffectNet/processed_csv_files/training_group_expva.csv"

    dataset = AffectnetPairExpression(data_root, path_to_csv, augment=True)
    # load data to PyTorch dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    #for epochs in range(1, 10):
    #    for i, data_dict in enumerate(dataloader):
    #        print(f'******{i}*********', data_dict['target'])

    visualise_AffectnetPair(dataset, batch_size=2)
    # y = total_pairs_AffectnetPair(dataset, batch_size=64)

