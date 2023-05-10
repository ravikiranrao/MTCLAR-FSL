from __future__ import print_function

from pathlib import Path
import torch
import torch.nn as nn

from .emonet import EmoNet


class SiameseNetEmoNetMLP(nn.Module):
    """Siamese network with EmoNet and MLP"""

    def __init__(self, emofan_n_emotions, is_pretrained, feat_fusion, num_neurons, dropout_rate, is_multi_task=False):
        super(SiameseNetEmoNetMLP, self).__init__()

        # Fusion technique
        self.feat_fusion = feat_fusion

        self.is_pretrained = is_pretrained

        # concatenate features, take absolute difference between the features,
        # or do elementwise multiplication of the features.
        assert self.feat_fusion in ["concat", "absolute", "multiply"]

        self.is_multi_task = is_multi_task
        # Number of neurons in the last layer
        self.num_neurons = num_neurons
        if self.is_multi_task:
            assert isinstance(self.num_neurons, tuple)
            num_neurons_fc = self.num_neurons[1]
        else:
            assert isinstance(self.num_neurons, int)
            num_neurons_fc = self.num_neurons

        self.emo_net = EmoNet(num_modules=2, n_expression=emofan_n_emotions, n_reg=2, n_blocks=4, attention=True,
                              temporal_smoothing=False)

        if self.is_pretrained:
            # Load weights to emonet
            state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{emofan_n_emotions}.pth')
            print(f'Loading EmoFAN from {state_dict_path}')
            state_dict = torch.load(str(state_dict_path))
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.emo_net.load_state_dict(state_dict, strict=False)

        # model.emo_fc_2[3] = nn.Linear(in_features=128, out_features=n_reg, bias=True)

        self.fc_in_features = self.emo_net.emo_fc_2[0].in_features  # [bs, 256]

        # get features before the last linear block
        # Refer emonet supplementary material for more info
        self.emo_net.emo_fc_2 = nn.Identity()    # Returns [bs, 256]

        if self.feat_fusion == "concat":
            feature_coeff = 2
        elif (self.feat_fusion == "absolute") or (self.feat_fusion == "multiply"):
            feature_coeff = 1
        else:
            raise ValueError

        self.dropout_rate = dropout_rate

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_in_features * feature_coeff, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128,
                      num_neurons_fc),
        )
        # initialize the weights
        self.fc.apply(self.init_weights)

        if self.is_multi_task:
            # Branch for multitask learning
            self.fc0 = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.fc_in_features * feature_coeff, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2048, 1024, bias=True),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(1024, 512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128,
                          self.num_neurons[0]),
            )
            self.fc0.apply(self.init_weights)

            self.fc2 = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.fc_in_features * feature_coeff, 2048, bias=True),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2048, 1024, bias=True),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(1024, 512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 128, bias=True),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128,
                          self.num_neurons[0]),
            )
            self.fc2.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward_once(self, x):
        output = self.emo_net(x)
        output = output['feature'].view(output['feature'].size()[0], -1)  # [bs, 256]
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.feat_fusion == "concat":
            # concatenate both images' features
            output = torch.cat((output1, output2), 1)
        elif self.feat_fusion == "absolute":
            # Take the absolute value between the two features
            output = torch.abs(output1 - output2)
        elif self.feat_fusion == "multiply":
            # Element wise multiplication
            output = output1 * output2
        else:
            raise ValueError

        if self.is_multi_task:
            # pass the concatenation to the linear layers
            # Similarity Head
            target = self.fc(output)

            # Delta valence Head
            target0 = self.fc0(output)

            # Delta Arousal Head
            target2 = self.fc2(output)

            return {'output1': output1, 'output2': output2, 'target': target, 'target0': target0,
                    'target2': target2
            }
        else:
            # pass the concatenation to the linear layers
            target = self.fc(output)

            return {'output1': output1, 'output2': output2, 'target': target}
