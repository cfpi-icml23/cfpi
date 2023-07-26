from collections import OrderedDict

import eztils.torch as ptu
import numpy as np
import torch
from eztils.torch import LayerNorm, ParallelLayerNorm, activation_from_string
from torch import nn
from torch.nn import functional as F


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def ident(x):
    return x


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=ident,
        hidden_init=fanin_init,
        b_init_value=0.0,
        layer_norm=False,
        dropout=False,
        dropout_kwargs=None,
        layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.layer_norm_kwargs = layer_norm_kwargs
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__(f"fc{i}", fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size, **layer_norm_kwargs)
                self.__setattr__(f"layer_norm{i}", ln)
                self.layer_norms.append(ln)

        self.dropout_kwargs = dropout_kwargs
        self.dropout = dropout
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            if self.dropout:
                F.dropout(h, **self.dropout_kwargs)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class MultiHeadedMlp(Mlp):
    """
                   .-> linear head 0
                  /
    input --> MLP ---> linear head 1
                  \
                   .-> linear head 2
    """

    def __init__(
        self,
        hidden_sizes,
        output_sizes,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activations=None,
        hidden_init=fanin_init,
        b_init_value=0.0,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=sum(output_sizes),
            input_size=input_size,
            init_w=init_w,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init,
            b_init_value=b_init_value,
            layer_norm=layer_norm,
            layer_norm_kwargs=layer_norm_kwargs,
        )
        self._splitter = SplitIntoManyHeads(
            output_sizes,
            output_activations,
        )

    def forward(self, input):
        flat_outputs = super().forward(input)
        return self._splitter(flat_outputs)


class ConcatMultiHeadedMlp(MultiHeadedMlp):
    """
    Concatenate inputs along dimension and then pass through MultiHeadedMlp.
    """

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class SplitIntoManyHeads(nn.Module):
    """
           .-> head 0
          /
    input ---> head 1
          \
           '-> head 2
    """

    def __init__(
        self,
        output_sizes,
        output_activations=None,
    ):
        super().__init__()
        if output_activations is None:
            output_activations = ["identity" for _ in output_sizes]
        else:
            if len(output_activations) != len(output_sizes):
                raise ValueError(
                    "output_activation and output_sizes must have " "the same length"
                )

        self._output_narrow_params = []
        self._output_activations = []
        for output_activation in output_activations:
            if isinstance(output_activation, str):
                output_activation = activation_from_string(output_activation)
            self._output_activations.append(output_activation)
        start_idx = 0
        for output_size in output_sizes:
            self._output_narrow_params.append((start_idx, output_size))
            start_idx = start_idx + output_size

    def forward(self, flat_outputs):
        pre_activation_outputs = tuple(
            flat_outputs.narrow(1, start, length)
            for start, length in self._output_narrow_params
        )
        outputs = tuple(
            activation(x)
            for activation, x in zip(self._output_activations, pre_activation_outputs)
        )
        return outputs


class ParallelMlp(nn.Module):
    """
    Efficient implementation of multiple MLPs with identical architectures.

           .-> mlp 0
          /
    input ---> mlp 1
          \
           '-> mlp 2

    See https://discuss.pytorch.org/t/parallel-execution-of-modules-in-nn-modulelist/43940/7
    for details

    The last dimension of the output corresponds to the MLP index.
    """

    def __init__(
        self,
        num_heads,
        input_size,
        output_size,  # per mlp
        hidden_sizes,
        hidden_activation="ReLU",
        output_activation="identity",
        dim=1,
        layer_norm=False,
        dropout=False,
        input_is_already_expanded=False,
    ):
        super().__init__()

        def create_layers():
            layers = []
            input_dim = input_size
            for i, hidden_size in enumerate(hidden_sizes):
                fc = nn.Conv1d(
                    in_channels=input_dim * num_heads,
                    out_channels=hidden_size * num_heads,
                    kernel_size=1,
                    groups=num_heads,
                )
                # fc.register_forward_hook(self.forward_hook(i))
                layers.append(fc)
                if isinstance(hidden_activation, str):
                    activation = activation_from_string(hidden_activation)
                else:
                    activation = hidden_activation
                layers.append(activation)

                if layer_norm:
                    ln = ParallelLayerNorm(num_heads, hidden_size)
                    layers.append(ln)
                    # ln.register_forward_hook(self.forward_hook(f"{i} ln"))

                if dropout:
                    drop = nn.Dropout(p=0.4)
                    layers.append(drop)

                input_dim = hidden_size

            last_fc = nn.Conv1d(
                in_channels=input_dim * num_heads,
                out_channels=output_size * num_heads,
                kernel_size=1,
                groups=num_heads,
            )
            layers.append(last_fc)

            if output_activation != "identity":
                if isinstance(output_activation, str):
                    activation = activation_from_string(output_activation)
                else:
                    activation = output_activation
                layers.append(activation)
            return layers

        self.network = nn.Sequential(*create_layers())
        self.num_heads = num_heads
        self.input_is_already_expanded = input_is_already_expanded
        self.dim = dim
        self.layer_norm = layer_norm
        # self.selected_out = OrderedDict()

    # def forward_hook(self, layer_name):
    #     def hook(module, input, output):
    #         self.selected_out[layer_name] = output

    #     return hook

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=self.dim)

        if not self.input_is_already_expanded:
            x = x.repeat(1, self.num_heads).unsqueeze(-1)
        flat = self.network(x)
        batch_size = x.shape[0]
        return flat.view(batch_size, -1, self.num_heads)

    @staticmethod
    def ensemble_to_individual(ens):  # ens: ParallelMlp
        ret = []
        layer_sizes = []
        for layer in ens.network:
            if isinstance(layer, nn.Conv1d):
                layer_sizes.append(
                    (
                        int(layer.in_channels / ens.num_heads),
                        int(layer.out_channels / ens.num_heads),
                    )
                )

        for i in range(ens.num_heads):
            mlp = ConcatMlp(
                hidden_sizes=[sz[1] for sz in layer_sizes[:-1]],
                output_size=layer_sizes[-1][-1],
                input_size=layer_sizes[0][0],
            )
            with torch.no_grad():
                constructed_state_dict = OrderedDict()

                ens_state_dict = ens.state_dict()
                for mlp_key, ens_key in zip(mlp.state_dict(), ens_state_dict):
                    tensor = ens_state_dict[ens_key].squeeze()
                    single_sz = int(tensor.shape[0] / ens.num_heads)
                    constructed_state_dict[mlp_key] = tensor[
                        single_sz * i : single_sz * (i + 1)
                    ]

                mlp.load_state_dict(constructed_state_dict)

            ret.append(mlp)
        return ret


class QuantileMlp(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        input_size,
        embedding_size=64,
        num_quantiles=8,
        layer_norm=True,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = ptu.from_numpy(np.arange(1, 1 + self.embedding_size))

    def forward(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output

    def get_tau_quantile(self, state, action, tau):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)
        with torch.no_grad():
            tau_pt = ptu.ones([state.shape[0], 1, 1]) * tau
            x = torch.cos(tau_pt * self.const_vec * np.pi)  # (N, 1, E)
            x = self.tau_fc(x)  # (N, 1, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, 1, C)
        h = self.merge_fc(h)  # (N, 1, C)
        output = self.last_fc(h).squeeze()  # (N, 1)
        return output

    def get_mean(self, state, action):
        """
        Calculate Quantile Mean in Batch (E(Z) = Q)
        tau: quantile fractions, (N, T)
        N = batch
        C = hidden sz (256)
        E = embedding sz (64)
        """
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        with torch.no_grad():
            presum_tau = ptu.zeros(state.shape[0], 32) + 1.0 / 32  # (N, 32)
            tau = torch.cumsum(presum_tau, dim=1)
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.0
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.0  # (N, 32)

            x = torch.cos(tau_hat.unsqueeze(-1) * self.const_vec * np.pi)  # (N, 32, E)
            x = self.tau_fc(x)  # (N, 32, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, 32, C)
        h = self.merge_fc(h)  # (N, 32, C)
        output = self.last_fc(h).squeeze()  # (N, 32) #! gets rid of C
        return output.mean(-1)  # (N,)
