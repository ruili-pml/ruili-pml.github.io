---
title: "hello world"
date: "2025-11-28"
tags: ["test"]
---
Test whether this setup works

## some heading..

test some equation

$$
s(\mathbf{a}_t) \triangleq
\sum_{i=1}^{n_{\text{init}}} \mathbf{a}_t[i]
+ \sum_{j=t-n_{\text{local}}}^{t} \mathbf{a}_t[j],
\quad t = T-k, \ldots, T.
$$


test some code

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```