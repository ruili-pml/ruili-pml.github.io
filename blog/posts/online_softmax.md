---
title: "Online Softmax: the math behind FlashAttention and PagedAttention"
date: "2025-11-30"
tags: ["Online Softmax", "FlashAttention", "PagedAttention"]
---

# Memory Bottleneck in Self-attention

Denote the query, key, and value matrices as  

$$ 
\mathbf{Q} \in \mathbb{R}^{T \times d}, \quad \mathbf{K} \in \mathbb{R}^{T \times d}, \quad \mathbf{V} \in \mathbb{R}^{T \times d}
$$

where $T$ is the number of tokens in the sequence and $d$ is the attention-head dimension.

In self-attention, the attention weights are
$$
\mathbf{A}
= \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)
\in \mathbb{R}^{T \times T}.
$$
and single-head attention output is
$$
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{T \times d}
$$

The most straightforward way to implement this is first construct $\mathbf{A}$, then do the matrix multiplication. 
For big models, this is problematic on two levels:

The "Too Big" Problem (FlashAttention)

The size of $\mathbf{A}$ grows quadratically. For a sequence length of $T = 32,000$ using FP16 precision, storing $\mathbf{A}$ requires $\approx 2 \text{GB}$ _per head_. Even if we had the capacity, the speed is bottlenecked by the time it takes to read and write this massive matrix to HBM (High Bandwidth Memory).

The "Fragmented" Problem (PagedAttention)

To construct $\mathbf{A}$ via a standard matrix multiplication, we generally need the full $\mathbf{K}$ matrix stored in a contiguous block of memory. During inference time with KV cache this can easily cause fragmentation.

Both scenarios force us to abandon the idea of computing the full Softmax at once. We need an approach that processes the input in chunks, computes local results, and then combining them to get the global exact answer.

This is where Online Softmax comes in.


# Closer Look at Softmax
Given an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_d] \in \mathbb{R}^{d \times 1}$, the standard Softmax is

$$
\text{Softmax}(\mathbf{x}) =
\left[
\frac{\exp(x_1)}{\sum_{j=1}^d \exp(x_j)},
\frac{\exp(x_2)}{\sum_{j=1}^d \exp(x_j)},
\ldots,
\frac{\exp(x_d)}{\sum_{j=1}^d \exp(x_j)}
\right].
$$

Since we exponentiate every element, large positive values in $\mathbf{x}$ can easily cause numerical overflow (resulting in `NaN`). Therefore, practical implementations use the numerically stable version by subtracting the maximum value from inputs:
$$
\mathrm{Softmax}(\mathbf{x}) = \left[\frac{\exp(x_1 - m)}{\sum_j \exp(x_j - m)}, \frac{\exp(x_2 - m)}{\sum_j \exp(x_j - m)}, \ldots, \frac{\exp(x_d - m)}{\sum_j \exp(x_j - m)} \right]
$$
where
$$
m = \max(x_1, \ldots, x_d).
$$

To compute this, the standard approach 

1. Find $m$.

2. Compute $\exp(\mathbf{x}-m)$ and the denominator $\ell$.

3. Divide to get the result.

```python
# Standard approach requires materializing the full vector
m = np.max(x)
e_x = np.exp(x - m) 
softmax = e_x / e_x.sum()
```

This requires materializing the entire vector `e_x` in memory. For huge sequence lengths, this is exactly what we want to avoid, that's where online Softmax comes in.

# Online Softmax
Online Softmax allows us to compute these values one element (or one chunk) at a time, updating statistics as we go.

In Softmax, we have two values that depends on the whole data:

- max $m = \max(x_1, \ldots, x_d)$
- denominator $\ell = \sum_{j=1}^d \exp(x_j - m)$

So in Online Softmax, we keep a running max $m_k$ and running denominator $\ell_k$, and update them as we traverse the data.

## Single-element Update
Let's start by doing update one element at a time.
### Update for the running max
What we have access to:

- Old Global State: $m_{k-1}, \ell_{k-1}$
- New Data: $x_k$

The new max is simply the larger of the current value $x_k$ and the previous max $m_{k-1}$:

$$m_k = \max(x_k,\, m_{k-1}).$$

### Update for the running denominator
We need to express the new sum $\ell_k$ using the old sum $\ell_{k-1}$ without re-summing the previous elements.

What we have access to:

- Old Global State: $m_{k-1}, \ell_{k-1}$
- New Global State: $m_k$
- New Data: $x_k$

$$
\begin{aligned}
\ell_k
&= \sum_{j=1}^{k} \exp(x_j - m_k) \\
&= \exp(x_k - m_k)
 + \sum_{j=1}^{k-1} \exp(x_j - m_k) \\
&= \exp(x_k - m_k)
 + \sum_{j=1}^{k-1} \exp(x_j - m_{k-1})\,\exp(m_{k-1} - m_k) \\
&= \exp(x_k - m_k)
 + \exp(m_{k-1} - m_k)\,\ell_{k-1}.
\end{aligned}
$$

## Chunk Update
In practice, we process data in chunks (blocks) to utilize GPU parallelism. Suppose we receive a chunk of data $x_{k}, \ldots, x_{k+r}$.

At time step $k$, what stored in memory is our previous statistics $(m_{k-1}, \ell_{k-1})$ and a new input chunk $x_{k}, \ldots, x_{k+r}$.

### Update for running max
We will first find the local max of the new chunk, let's call it $\widetilde{m}$.
Then update the global max.

What we have access to:

- Old Global State: $m_{k-1}, \ell_{k-1}$
- New Data: $x_{k}, \ldots, x_{k+r}$

$$
\widetilde{m}_{k+r} = \max(x_k,\ldots,x_{k+r}) \\
m_{k+r} = \max\!\big(m_{k-1},\, \widetilde{m}_{k+r}\big).
$$

### Update for the running denominator
The idea is similar to single-element update.
We express $\ell_{k+r}$ in a form of values we currently have access to.

What we have access to:

- Old Global State: $m_{k-1}, \ell_{k-1}$
- New Local State: $\widetilde{m}_{k+r}$
- New Global State: $m_{k+r}$
- New Data: $x_{k}, \ldots, x_{k+r}$

$$
\begin{aligned}
\ell_{k+r}
&= \sum_{j=1}^{k+r} \exp(x_j - m_{k+r}) \\
&= \sum_{j=1}^{k-1} \exp(x_j - m_{k+r}) 
   \;+\; \sum_{i=k}^{k+r} \exp(x_i - m_{k+r}) \\
&= \sum_{j=1}^{k-1} \exp\!\big(x_j - m_{k+r} + m_{k-1}-m_{k-1}\big)
   \;+\; \sum_{i=k}^{k+r} \exp(x_i - m_{k+r}) \\
&= \exp(m_{k-1}-m_{k+r}) \sum_{j=1}^{k-1} \exp(x_j - m_{k-1})
   \;+\; \sum_{i=k}^{k+r} \exp(x_i - m_{k+r}) \\
&= \exp(m_{k-1}-m_{k+r})\,\ell_{k-1}
   \;+\; \sum_{i=k}^{k+r} \exp(x_i - m_{k+r}) \\
&= \exp(m_{k-1}-m_{k+r})\,\ell_{k-1}
 + \exp(\widetilde{m}_{k+r} - m_{k+r})
   \sum_{i=k}^{k+r} \exp(x_i - \widetilde{m}_{k+r})
\end{aligned}
$$



# Online Attention Output
In the end we care about the weighted sum of values, let's look at how to compute this in an online manner.

## Single-element Update

What we have access to:

- Old Global State: $m_{k-1}$, $\ell_{k-1}$, $\mathbf{o}_{k-1}$
- New Global State: $m_k$, $\ell_k$
- New Data: $x_k$, $\mathbf{v}_k$

$$
\begin{aligned}
\mathbf{o}_{k}
&= \sum_{j=1}^{k} 
   \frac{\exp(x_j - m_{k})}{\ell_{k}}\,\mathbf{v}_j
\\[4pt]
&= \frac{\exp(x_1 - m_{k})}{\ell_{k}}\,\mathbf{v}_1
 + \frac{\exp(x_2 - m_{k})}{\ell_{k}}\,\mathbf{v}_2
 + \cdots
 + \frac{\exp(x_{k} - m_{k})}{\ell_{k}}\,\mathbf{v}_{k}
\quad \text{(same for every sum term)} 
\\[6pt]
&= \frac{\exp(-m_{k})}{\ell_{k}}
   \sum_{j=1}^{k} \exp(x_j)\,\mathbf{v}_j
\\[6pt]
&= \frac{\exp(-m_{k})}{\ell_{k}}
   \left(
     \sum_{j=1}^{k-1} \exp(x_j)\,\mathbf{v}_j
     \;+ \exp(x_k)\,\mathbf{v}_k
   \right)
\\[6pt]
&= \frac{\exp(-m_{k})}{\ell_{k}}
   \left(
     \sum_{j=1}^{k-1} 
       \frac{\exp(x_j)\exp(-m_{k-1})}{\ell_{k-1}}
       \cdot \frac{\ell_{k-1}}{\exp(-m_{k-1})}\,\mathbf{v}_j
     \;+\; \exp(x_k)\,\mathbf{v}_k
   \right)
\\[6pt]
&=  \frac{\exp(-m_{k})}{\ell_{k}}  \exp(x_k)\,\mathbf{v}_k
  + \frac{\exp(-m_{k})\,\ell_{k-1}}{\ell_{k}\,\exp(-m_{k-1})}\,
    \underbrace{\sum_{j=1}^{k-1} 
      \frac{\exp(x_j - m_{k-1})}{\ell_{k-1}}\,\mathbf{v}_j}_{\mathbf{o}_{k-1}}
\\
&= \frac{\exp(x_k - m_k)}{\ell_k}\mathbf{v}_k
 + \frac{\exp(-m_k)\,\ell_{k-1}}{\ell_k\,\exp(-m_{k-1})}\,\mathbf{o}_{k-1}
\end{aligned}
$$


## Chunk Update
What we have access to:

- Old Global State: $m_{k-1}, \ell_{k-1}$
- New Local State: $\widetilde{m}_{k+r}$
- New Global State: $m_{k+r}$, $\ell_{k+r}$
- New Data: $\mathbf{x}_{k}, \ldots, \mathbf{x}_{k+r}$, $\mathbf{v}_{k}, \ldots, \mathbf{v}_{k+r}$


$$
\begin{aligned}
\mathbf{o}_{k+r}
&= \sum_{j=1}^{k+r} 
   \frac{\exp(x_j - m_{k+r})}{\ell_{k+r}}\,\mathbf{v}_j
\\[4pt]
&= \frac{\exp(x_1 - m_{k+r})}{\ell_{k+r}}\,\mathbf{v}_1
 + \frac{\exp(x_2 - m_{k+r})}{\ell_{k+r}}\,\mathbf{v}_2
 + \cdots
 + \frac{\exp(x_{k+r} - m_{k+r})}{\ell_{k+r}}\,\mathbf{v}_{k+r}
\\[6pt]
&= \frac{\exp(-m_{k+r})}{\ell_{k+r}}
   \sum_{j=1}^{k+r} \exp(x_j)\,\mathbf{v}_j
\\[6pt]
&= \frac{\exp(-m_{k+r})}{\ell_{k+r}}
   \left(
     \sum_{j=1}^{k-1} \exp(x_j)\,\mathbf{v}_j
     \;+\; \sum_{i=k}^{k+r} \exp(x_i)\,\mathbf{v}_i
   \right)
\\[6pt]
&= \frac{\exp(-m_{k+r})}{\ell_{k+r}}
   \left(
     \sum_{j=1}^{k-1} 
       \frac{\exp(x_j)\exp(-m_{k-1})}{\ell_{k-1}}
       \cdot \frac{\ell_{k-1}}{\exp(-m_{k-1})}\,\mathbf{v}_j
     \;+\; \sum_{i=k}^{k+r} \exp(x_i)\,\mathbf{v}_i
   \right)
\\[6pt]
&= \frac{\exp(-m_{k+r})\,\ell_{k-1}}{\ell_{k+r}\,\exp(-m_{k-1})}\,
    \mathbf{o}_{k-1}
   \;+\; \sum_{i=k}^{k+r} \frac{\exp(x_i - m_{k+r})}{\ell_{k+r}}\,\mathbf{v}_i 
\\[6pt]
&= \frac{\exp(-m_{k+r})\,\ell_{k-1}}{\ell_{k+r}\,\exp(-m_{k-1})}\, \mathbf{o}_{k-1}
 + \exp(\widetilde{m}_{k+r} - m_{k+r})
   \sum_{i=k}^{k+r} \frac{\exp(x_i - \widetilde{m}_{k+r})}{\ell_{k+r}}\,\mathbf{v}_i
\end{aligned}
$$


# Summary
In short, Online Softmax gives us a way to compute attention output **exactly** without ever constructing the full $T \times T$ attention weight matrix. 
Combined with a few additional scheduling and memory-management tricks, this is the core idea that makes FlashAttention and PagedAttention capable of handling long sequences efficiently.


Denote the $i$-th attention score as $x_i$. The update rules are 

Old running results:  
$m_{k-1},\ \ell_{k-1},\ \mathbf{o}_{k-1}$

New data:  
$x_k,\ldots,x_{k+r},\ \mathbf{v}_k,\ldots,\mathbf{v}_{k+r}$

Update rules:

$$
\widetilde{m}_{k+r} = \max(x_k,\ldots,x_{k+r})
$$

$$
m_{k+r} = \max(m_{k-1},\, \widetilde{m}_{k+r})
$$

$$
\ell_{k+r}
= \exp(m_{k-1}-m_{k+r})\,\ell_{k-1}
 + \exp(\widetilde{m}_{k+r} - m_{k+r})
   \sum_{i=k}^{k+r} \exp(x_i - \widetilde{m}_{k+r})
$$

$$
\mathbf{o}_{k+r}
= \frac{\exp(-m_{k+r})\,\ell_{k-1}}{\ell_{k+r}\,\exp(-m_{k-1})}\mathbf{o}_{k-1}
 + \exp(\widetilde{m}_{k+r} - m_{k+r})
   \sum_{i=k}^{k+r} \frac{\exp(x_i - \widetilde{m}_{k+r})}{\ell_{k+r}}\,\mathbf{v}_i
$$

This matches the expression in FlashAttention paper

$$
\widetilde{m}_{k+r} = \max(x_k,\ldots,x_{k+r})
$$

$$
\widetilde{\ell}_{k+r} = \sum_{i=k}^{k+r} \exp(x_i - \widetilde{m}_{k+r})
$$

$$
m_{k+r} = \max(m_{k-1},\, \widetilde{m}_{k+r})
$$

$$
\ell_{k+r}
= \exp(m_{k-1}-m_{k+r})\,\ell_{k-1}
 + \exp(\widetilde{m}_{k+r} - m_{k+r})\,\widetilde{\ell}_{k+r}
$$

$$
\mathbf{o}_{k+r}
= \frac{1}{\ell_{k+r}}
\left(
\ell_{k-1}\exp(m_{k-1}-m_{k+r})\,\mathbf{o}_{k-1}
+ \exp(\widetilde{m}_{k+r}-m_{k+r})
  \sum_{i=k}^{k+r} \exp(x_i - \widetilde{m}_{k+r})\,\mathbf{v}_i
\right)
$$

# Reference 

This blog post helped me a lot on understanding how online update works: 
https://alvinwan.com/how-flash-attention-works/

FlashAttention paper:
https://arxiv.org/abs/2205.14135