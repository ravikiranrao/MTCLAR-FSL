import os
import pandas as pd
import numpy as np
import random

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

pd.set_option('mode.chained_assignment', None)


class AfewvaPairExpression(Dataset):
    def __init__(self, data_root, path_to_csv, n_expression=7, image_shape=(3, 256, 256), augment=False, dataset_len=-1,
                 verbose=0):
        super(AfewvaPairExpression, self).__init__()

        # root folder containing all the images
        self.data_root = data_root

        # get dataset
        self.data_df = pd.read_csv(path_to_csv, dtype={'video_id': str, 'frame_num': str, 'actor': str})

        # Extract only the relevant columns
        self.data_df = self.data_df[['video_id', 'frame_num', 'actor', 'valence', 'arousal', 'expression']]
        # data_df = data_df.sample(250, random_state=5, ignore_index=True)

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
        # return int(((self.data_df.shape[0]) * (self.data_df.shape[0]-1))/2) This is too big
        if self.dataset_len < 0:
            # if sf.dataset_len is -5, then 5 * self.data_df.shape[0]
            return abs(self.dataset_len) * self.data_df.shape[0]
        else:
            return self.dataset_len

    def __getitem__(self, index):
        # pick some random class for the first image
        selected_class = random.randint(0, self.n_expression - 1)

        # pick a random index for the first image in the grouped indices based on the label of the class
        random_index_1 = random.choice(self.group_dict[selected_class])

        # get the first image
        image_1_path = os.path.join(self.data_df.iloc[random_index_1].video_id,
                                    self.data_df.iloc[random_index_1].frame_num)

        # same class
        if index % 2 == 0:

            # pick a random index for the second image
            random_index_2 = random.choice(self.group_dict[selected_class])

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.choice(self.group_dict[selected_class])

            # get the second image
            image_2_path = os.path.join(self.data_df.iloc[random_index_2].video_id,
                                        self.data_df.iloc[random_index_2].frame_num)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.long)

        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, self.n_expression - 1)

            # ensure that the class of the second image is not same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, self.n_expression - 1)

            # pick a random index for the second image
            random_index_2 = random.choice(self.group_dict[other_selected_class])

            # get the second image
            image_2_path = os.path.join(self.data_df.iloc[random_index_2].video_id,
                                        self.data_df.iloc[random_index_2].frame_num)

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


class AfewvaAnchorPair(Dataset):
    """Im1 -> Anchor Im (same actor)"""

    def __init__(self, data_root, path_to_csv, k_shot=1, n_expression=7, image_shape=(3, 256, 256), augment=False,
                 verbose=0):
        super(AfewvaAnchorPair, self).__init__()

        # root folder containing all the images
        self.data_root = data_root

        # get dataset
        data_df = pd.read_csv(path_to_csv, dtype={'video_id': str, 'frame_num': str, 'actor': str})

        # Extract only the relevant columns
        self.data_df = data_df[['video_id', 'frame_num', 'actor', 'valence', 'arousal', 'expression']]
        # self.data_df = self.data_df.head(2000)

        # Rescale all the valence ar values from [-10, 10] to [-1, 1]
        self.data_df['valence'] = self.data_df['valence'] / 10
        self.data_df['arousal'] = self.data_df['arousal'] / 10

        # Remove rows with Nan values
        self.data_df = self.data_df.dropna(how='any', axis=0)

        # reset index
        self.data_df.reset_index(inplace=True)

        # Number of examples to consider in each class
        self.k_shot = k_shot

        # get only one sample per class
        # self.data_df.groupby('expression').first().reset_index()
        # self.data_df = pd.DataFrame(self.data_df.groupby('expression').head(2).reset_index(drop=True))
        #self.actor_df = pd.DataFrame(self.data_df.groupby('actor').head(self.k_shot).reset_index(drop=True))
        #self.actor_df = pd.DataFrame(self.data_df.groupby('actor').sample(self.k_shot).reset_index(drop=True))
        #self.actor_df = pd.DataFrame(self.data_df.groupby('video_id').head(self.k_shot).reset_index(drop=True))
        #self.actor_df = pd.DataFrame(self.data_df.groupby('video_id').sample(self.k_shot).reset_index(drop=True))
        self.actor_df = pd.DataFrame(self.data_df.sort_values(by=['video_id', 'frame_num'],
                                                              ascending=True).groupby('video_id',
                                                                                      as_index=False).nth(range(0, 1000, 20)).reset_index(drop=True))

        # Anchor df from the train set.
        # Get images for the support set from the training set
        # create bins and pick n samples from each bin
        #num_bins = 10 # -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
        #bins = pd.IntervalIndex.from_tuples([(-1., -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5),
        #                                     (-0.5, -0.4), (-0.4, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0),
        #                                     (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
        #                                     (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.)])
        #num_items_per_bin = 10
        #path_to_csv_train = "data/AfewVA/processed_csv_files/group_fold_1_training.csv"

        #actor_df = pd.read_csv(path_to_csv_train, dtype={'video_id': str, 'frame_num': str, 'actor': str})
        # Rescale all the valence ar values from [-10, 10] to [-1, 1]
        #actor_df['valence'] = actor_df['valence'] / 10
        #actor_df['arousal'] = actor_df['arousal'] / 10
        # crete bins and assign them to bin column
        #actor_df['bin'] = pd.cut(actor_df['valence'], num_bins, labels=False)
        #self.actor_df = actor_df.groupby('bin').apply(lambda x: x.sample(num_items_per_bin))
        #self.actor_df = self.actor_df.drop(columns=['bin'])

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
        image_1_path = os.path.join(self.data_df.iloc[index].video_id, self.data_df.iloc[index].frame_num)
        #a = self.actor_df.loc[self.actor_df.actor == self.data_df.iloc[index].actor]
        #a = a.sample(1)
        #a = self.actor_df.loc[self.actor_df.actor != self.data_df.iloc[index].actor]
        #a = a.sample(frac=1)
        #a = self.actor_df.loc[self.actor_df.video_id != self.data_df.iloc[index].video_id]
        #a = self.data_df.sample(1)
        a = self.actor_df.loc[self.actor_df.video_id == self.data_df.iloc[index].video_id]
        a = a[a.frame_num <= self.data_df.iloc[index].frame_num]
        a = a.sort_values(by='frame_num', ascending=False)
        #a = self.actor_df.sample(1)

        # Anchor image
        image_2_path = os.path.join(a.video_id.values[0], a.frame_num.values[0])

        # Load both the images
        image1 = Image.open(os.path.join(self.data_root, image_1_path)).convert("RGB")
        image2 = Image.open(os.path.join(self.data_root, image_2_path)).convert("RGB")

        # Transform
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # get other labels
        valence = [self.data_df.iloc[index].valence, a.valence.values[0]]

        arousal = [self.data_df.iloc[index].arousal, a.arousal.values[0]]

        random_index = [index, index]

        categories = [self.data_df.iloc[index].expression, a.expression.values[0]]

        actor = [self.data_df.iloc[index].actor, a.actor.values[0]]
        video_id = [self.data_df.iloc[index].video_id, a.video_id.values[0]]
        # print(f'Index: {index}, image_1_path: {image_1_path}, image2_path: {image_2_path}')

        return dict(image1=image1, image2=image2, target=torch.FloatTensor(1), valence=torch.FloatTensor(valence),
                    arousal=torch.FloatTensor(arousal), random_index=torch.FloatTensor(random_index),
                    delta_valence=torch.tensor(valence[0] - valence[1], dtype=torch.float),
                    delta_arousal=torch.tensor(arousal[0] - arousal[1], dtype=torch.float),
                    image_1_path=image_1_path, image_2_path=image_2_path,
                    categories=torch.FloatTensor(categories),
                    actor=actor, video_id=video_id)


# Data loader for getting an image, as opposed to *pair* of images.
class AfewvaIdenticalPair(Dataset):
    """This class is used to get same images as a pair. This is used when getting features from the trained model
    and retraining features using kNN to predict categorical classes."""

    def __init__(self, data_root, path_to_csv, image_shape=(3, 256, 256), augment=False, verbose=0):
        super(AfewvaIdenticalPair, self).__init__()

        # root folder containing all the images
        self.data_root = data_root

        # get dataset
        data_df = pd.read_csv(path_to_csv, dtype={'video_id': str, 'frame_num': str, 'actor': str})

        # Extract only the relevant columns
        self.data_df = data_df[['video_id', 'frame_num', 'actor', 'valence', 'arousal', 'expression']]
        # self.data_df = self.data_df.head(2000)

        # Rescale all the valence ar values from [-10, 10] to [-1, 1]
        self.data_df['valence'] = self.data_df['valence'] / 10
        self.data_df['arousal'] = self.data_df['arousal'] / 10

        # reset index
        self.data_df.reset_index(inplace=True)

        # get only one sample per class
        # self.data_df.groupby('expression').first().reset_index()
        # self.data_df = pd.DataFrame(self.data_df.groupby('expression').head(2).reset_index(drop=True))

        # input image size to the neural network
        self.image_shape = image_shape

        # Augment
        self.augment = augment

        if self.augment:
            # If images are to be augmented, add extra operations for it (first two).

            self.transform = transforms.Compose([
                transforms.Resize((self.image_shape[1] + 32, self.image_shape[2] + 32)),
                transforms.RandomAffine(degrees=(20, 50), translate=(0.2, 0.4), scale=(0.8, 1.2), shear=0.1),
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
        return self.data_df.shape[0]

    def __getitem__(self, index):
        index = index % self.data_df.shape[0]
        # get the first image
        image_1_path = os.path.join(self.data_df.iloc[index].video_id, self.data_df.iloc[index].frame_num)

        # Load both the images
        image1 = Image.open(os.path.join(self.data_root, image_1_path)).convert("RGB")

        # Transform
        if self.transform:
            image1 = self.transform(image1)

        # get other labels
        valence = [self.data_df.iloc[index].valence, self.data_df.iloc[index].valence]

        arousal = [self.data_df.iloc[index].arousal, self.data_df.iloc[index].arousal]

        random_index = [index, index]

        categories = [self.data_df.iloc[index].expression, self.data_df.iloc[index].expression]

        return dict(image1=image1, image2=image1, target=torch.FloatTensor(1), valence=torch.FloatTensor(valence),
                    arousal=torch.FloatTensor(arousal), random_index=torch.FloatTensor(random_index),
                    delta_valence=torch.tensor(valence[0] - valence[1], dtype=torch.float),
                    delta_arousal=torch.tensor(arousal[0] - arousal[1], dtype=torch.float),
                    image_1_path=image_1_path, image_2_path=image_1_path,
                    categories = torch.FloatTensor(categories))


# poor man's dataset
class AfewvaFeature(Dataset):
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
        return self.Feats.shape[0]

    def __getitem__(self, idx):
        """Generates one sample of data"""
        return dict(feats=self.Feats[idx, :], categories=int(self.categories[idx, ]),
                    valence=self.valence[idx, ], arousal=self.arousal[idx, ])


def visualise_AfewvaPair(dataset, batch_size=1):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_dict = next(iter(dataloader))

    # plot data
    # To have dark background
    # plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Visualise first sample in the batch
    axes[0].imshow(data_dict["image1"][0, :, :, :].permute(1, 2, 0))
    axes[0].set_title(f'Valence: {data_dict["valence"][0, 0]:.2f}'
                      f'\nArousal: {data_dict["arousal"][0, 0]:.2f}'
                      f'\n{data_dict["image_1_path"]}')
    axes[1].imshow(data_dict["image2"][0, :, :, :].permute(1, 2, 0))
    axes[1].set_title(f'Valence: {data_dict["valence"][0, 1]:.2f}'
                      f'\nArousal: {data_dict["arousal"][0, 1]:.2f}'
                      f'\n{data_dict["image_2_path"]}')
    #fig.suptitle(f'Similarity Label: {int(data_dict["target"][0])}')
    plt.show()
    return


if __name__ == "__main__":

    data_root = "../../data/AfewVA/Face_Raw_Data4"
    path_to_csv = "../../data/AfewVA/processed_csv_files/group_fold_1_test.csv"

    #dataset = AfewvaIdenticalPair(data_root, path_to_csv, augment=True)
    dataset = AfewvaAnchorPair(data_root, path_to_csv, augment=True, k_shot=1)
    # load data to PyTorch dataloader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

    for epochs in range(1, 2):
        for i, data_dict in enumerate(dataloader):
            continue
    for i in range(10):
        visualise_AfewvaPair(dataset, batch_size=2)
