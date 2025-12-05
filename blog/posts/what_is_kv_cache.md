---
title: "What is KV Cache? A Clear, Step-by-Step Explanation"
date: "2025-12-5"
tags: ["KV cache", "LLM Inference"]
---

KV cache (Key–Value cache) is an inference-time optimization used in Large Language Models (LLMs) to speed up generation. The core idea is this: instead of recomputing the **keys** and **values** for all previous tokens at every step, we will cache the computed keys and values and then **reuse** them during decoding.

Two properties of the usual transformer setup make this possible:

- **Causal masking** – token $t$ is only allowed to attend to tokens at positions $\leq t$, so once we have the keys and values for earlier tokens, they never change.
- **Next-token prediction** – the model only needs the representation of the *last* token to predict the next one, so we don’t need to recompute outputs for the whole sequence each time.

KV cache saves a huge amount of computation, especially for long prompts.

To understand why KV cache makes sense, the easiest way, in my opinion, is to write out the computation for a few tokens and see what actually happens.
Once the equations are laid out, it becomes obvious that certain things are being recomputed, and that we can save compute by storing them.

We'll consider a single-layer LLM with $H$ attention heads.  
For a deeper LLM, the logic is exactly the same, just applied layer by layer.

At inference time there are usually two stages:

1. **Prefill**: process the prompt in sequence.
2. **Autoregressive generation**: generate one token at a time and append it to the input.

---
# Two stage generation

## Step 1: Processing the Prompt (Prefill)

For simplicity, let's say the tokenised prompt has two tokens. So the input is

$$
\boldsymbol{X} =
\begin{bmatrix}
\boldsymbol{x}_1 \\[1em]
\boldsymbol{x}_2 \\
\end{bmatrix}
$$

Here, $\boldsymbol{x}_t$ is the embedding (or hidden state) of token $t$.

### RMSNorm

The first step is normalization. 
In most modern LLMs (like Llama), RMSNorm is applied on a _per-token level_. This means the normalization of $\boldsymbol{x}_1$ does not depend on $\boldsymbol{x}_2$.

$$
\widetilde{\boldsymbol{X}} =
\begin{bmatrix}
\operatorname{norm}\big(\boldsymbol{x}_1\big) \\[1em]
\operatorname{norm}\big(\boldsymbol{x}_2\big) \\
\end{bmatrix}
=
\begin{bmatrix}
\widetilde{\boldsymbol{x}}_1 \\[1em]
\widetilde{\boldsymbol{x}}_2\\
\end{bmatrix}
$$

So far, there is _no interaction between tokens_. Each token is just scaled and shifted on its own.
In other words, nothing here will ever depend on the rest of the sequence.

Next we enter the parallel computation of each single-head attention.
This is where the interaction between tokens happens, and it’s also where KV cache will later save us the most work.

### Single head attention of head $h$
For each head $h$, we first compute the query, key and value through linear projections.

$$
\boldsymbol{Q}^{(h)} =
\begin{bmatrix}
\widetilde{\boldsymbol{x}}_1 \boldsymbol{W}^{(h)}_{Q} \\[0.8em]
\widetilde{\boldsymbol{x}}_2 \boldsymbol{W}^{(h)}_{Q}
\end{bmatrix}
=
\begin{bmatrix}
\boldsymbol{q}^{(h)}_1 \\[0.8em]
\boldsymbol{q}^{(h)}_2
\end{bmatrix}
,\quad
\boldsymbol{K}^{(h)} =
\begin{bmatrix}
\widetilde{\boldsymbol{x}}_1 \boldsymbol{W}^{(h)}_{K} \\[0.8em]
\widetilde{\boldsymbol{x}}_2 \boldsymbol{W}^{(h)}_{K}
\end{bmatrix}
=
\begin{bmatrix}
\boldsymbol{k}^{(h)}_1 \\[0.8em]
\boldsymbol{k}^{(h)}_2
\end{bmatrix},\quad
\boldsymbol{V}^{(h)} =
\begin{bmatrix}
\widetilde{\boldsymbol{x}}_1 \boldsymbol{W}^{(h)}_{V} \\[0.8em]
\widetilde{\boldsymbol{x}}_2 \boldsymbol{W}^{(h)}_{V}
\end{bmatrix}
=
\begin{bmatrix}
\boldsymbol{v}^{(h)}_1 \\[0.8em]
\boldsymbol{v}^{(h)}_2
\end{bmatrix}.
$$

Here, the dependence is still local:
- $\boldsymbol{q}^{(h)}_t$ depends only on $\widetilde{\boldsymbol{x}}_t$ and $\boldsymbol{W}^{(h)}_Q$  
- $\boldsymbol{k}^{(h)}_t$ depends only on $\widetilde{\boldsymbol{x}}_t$ and $\boldsymbol{W}^{(h)}_K$  
- $\boldsymbol{v}^{(h)}_t$ depends only on $\widetilde{\boldsymbol{x}}_t$ and $\boldsymbol{W}^{(h)}_V$

Nothing here mixes information across different tokens yet; we are still just reshaping each token independently into Q/K/V space.

Next, we calculate the attention weights and single-head output.
This is where the mixing begins: each query now “looks at” all previous keys and decides how much to mix from their values.
 
The masked attention scores are

$$
\boldsymbol{S}^{(h)} =
\begin{bmatrix}
\boldsymbol{q}^{(h)}_1 \cdot \boldsymbol{k}^{(h)}_1 & -\infty \\[0.8em]
\boldsymbol{q}^{(h)}_2 \cdot \boldsymbol{k}^{(h)}_1 & \boldsymbol{q}^{(h)}_2 \cdot \boldsymbol{k}^{(h)}_2
\end{bmatrix}.
$$

With a **causal mask**, we ensure that each token cannot see future information. As we will see in a minute, **this also ensures that the hidden states of past tokens will never change when new tokens arrives**.

By applying a row-wise softmax, we get the attention weights.

$$
\boldsymbol{A}^{(h)} =
\begin{bmatrix}
\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_1\cdot \boldsymbol{k}^{(h)}_1\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_1\cdot \boldsymbol{k}^{(h)}_1\big) } &
0 \\[2.0em]
\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_1\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_2\big) } &
\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_2\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_2\big) }
\end{bmatrix}.
$$

Then, the output of each head is

$$
\begin{bmatrix}
\boldsymbol{o}^{(h)}_1 \\[0.8em]
\boldsymbol{o}^{(h)}_2
\end{bmatrix}
=
\begin{bmatrix}
A^{(h)}_{11} \boldsymbol{v}^{(h)}_1 \\[0.8em]
A^{(h)}_{21} \boldsymbol{v}^{(h)}_1 +
A^{(h)}_{22} \boldsymbol{v}^{(h)}_2
\end{bmatrix}.
$$

### Multi head attention output

Multi-head attention simply concatenates the outputs of all heads and mixes them with a final output projection.  
There is no information exchange between tokens here.

$$
\begin{bmatrix}
\boldsymbol{o}_1^{\text{Attn}} \\[1em]
\boldsymbol{o}_2^{\text{Attn}} \\
\end{bmatrix}=
\begin{bmatrix}
\boldsymbol{o}_1^{(1)} & \boldsymbol{o}_1^{(2)} & \ldots &  \boldsymbol{o}_1^{(H)}\\[1em]
\boldsymbol{o}_2^{(1)} & \boldsymbol{o}_2^{(2)} &  \ldots  &\boldsymbol{o}_2^{(H)}\\
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{W}_O^{(1)} \\[1em]
\boldsymbol{W}_O^{(2)} \\[1em]
\vdots \\
\boldsymbol{W}_O^{(H)} \\
\end{bmatrix}=
\begin{bmatrix}
\sum_{h=1}^H \boldsymbol{o}_1^{(h)}\boldsymbol{W}_O^{(h)}  \\[1em]
\sum_{h=1}^H \boldsymbol{o}_2^{(h)}\boldsymbol{W}_O^{(h)}\\
\end{bmatrix}
$$

### residual connection

The residual connection adds back the original token representations:

$$
\begin{bmatrix}
\boldsymbol{x}^{\text{MLP}}_1 \\[1em]
\boldsymbol{x}^{\text{MLP}}_2\\
\end{bmatrix}=
\begin{bmatrix}
\boldsymbol{o}_1^{\text{Attn}} + \boldsymbol{x}_1\\[1em]
\boldsymbol{o}_2^{\text{Attn}} + \boldsymbol{x}_2 \\
\end{bmatrix}
$$

### MLP

MLP process each token **independently**, so there is no information exchange between tokens.

$$
\begin{bmatrix}
\boldsymbol{o}_1^{\text{MLP}} \\[1em]
\boldsymbol{o}_2^{\text{MLP}} \\
\end{bmatrix}=
\begin{bmatrix}
\text{MLP}\Big(\text{norm}(\boldsymbol{x}^{\text{MLP}}_1) \Big) \\[1em]
\text{MLP}\Big(\text{norm}(\boldsymbol{x}^{\text{MLP}}_2) \Big) \\
\end{bmatrix}
$$


### Read out layer (predicting token 3)

Finally, the model predicts the *next* token, $\boldsymbol{x}_3$, based on the hidden states of the last token ($\boldsymbol{x}_2$):

$$
\boldsymbol{x}_3 = \text{read out} \left(\text{norm}(\boldsymbol{o}_2^{\text{MLP}} + \boldsymbol{x}^{\text{MLP}}_2) \right)
$$


## Step 2: Generation

So far, we have only processed the prompt, produced a distribution over token 3, and (conceptually) sampled $\boldsymbol{x}_3$ from it.
Now we append $\boldsymbol{x}_3$ into the input and continue the generation. This is where the redundancy happens.

Let’s first write out the naive computation, where we simply pretend this is a fresh 3-token input and redo everything from scratch.
I have hightlighted in blue the exact computation that we already performed during the prefill phase.

We feed in all three tokens:

$$
\boldsymbol{X} =
\begin{bmatrix}
\boldsymbol{x}_1 \\[1em]
\boldsymbol{x}_2 \\[1em]
\boldsymbol{x}_3
\end{bmatrix}
$$

### RMSNorm

$$
\widetilde{\boldsymbol{X}} =
\begin{bmatrix}
\textcolor{blue}{\operatorname{norm}\big(\boldsymbol{x}_1\big)} \\[1em]
\textcolor{blue}{\operatorname{norm}\big(\boldsymbol{x}_2\big)} \\[1em]
\operatorname{norm}\big(\boldsymbol{x}_3\big)
\end{bmatrix}
=
\begin{bmatrix}
\textcolor{blue}{\widetilde{\boldsymbol{x}}_1} \\[1em]
\textcolor{blue}{\widetilde{\boldsymbol{x}}_2}\\[1em]
\widetilde{\boldsymbol{x}}_3
\end{bmatrix}
$$

### Single head attention of head $h$
Compute query, key and values

$$
\boldsymbol{Q}^{(h)} =
\begin{bmatrix}
\textcolor{blue}{\widetilde{\boldsymbol{x}}_1 \boldsymbol{W}^{(h)}_{Q}} \\[1em]
\textcolor{blue}{\widetilde{\boldsymbol{x}}_2\boldsymbol{W}^{(h)}_{Q}} \\[1em]
\widetilde{\boldsymbol{x}}_3\boldsymbol{W}^{(h)}_{Q}
\end{bmatrix}
=
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{q}^{(h)}_1} \\[1em]
\textcolor{blue}{\boldsymbol{q}^{(h)}_2} \\[1em]
\boldsymbol{q}^{(h)}_3
\end{bmatrix}, \quad
\boldsymbol{K}^{(h)} =
\begin{bmatrix}
\textcolor{blue}{\widetilde{\boldsymbol{x}}_1 \boldsymbol{W}^{(h)}_{K}} \\[1em]
\textcolor{blue}{\widetilde{\boldsymbol{x}}_2\boldsymbol{W}^{(h)}_{K}} \\[1em]
\widetilde{\boldsymbol{x}}_3\boldsymbol{W}^{(h)}_{K}
\end{bmatrix}
=
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{k}^{(h)}_1} \\[1em]
\textcolor{blue}{\boldsymbol{k}^{(h)}_2} \\[1em]
\boldsymbol{k}^{(h)}_3
\end{bmatrix}
, \quad
\boldsymbol{V}^{(h)} =
\begin{bmatrix}
\textcolor{blue}{\widetilde{\boldsymbol{x}}_1 \boldsymbol{W}^{(h)}_{V}} \\[1em]
\textcolor{blue}{\widetilde{\boldsymbol{x}}_2 \boldsymbol{W}^{(h)}_{V}} \\[1em]
\widetilde{\boldsymbol{x}}_3 \boldsymbol{W}^{(h)}_{V}
\end{bmatrix}
=
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{v}^{(h)}_1} \\[1em]
\textcolor{blue}{\boldsymbol{v}^{(h)}_2} \\[1em]
\boldsymbol{v}^{(h)}_3
\end{bmatrix}
$$

Attention scores

$$
\boldsymbol{S}^{(h)} =
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{q}^{(h)}_1 \cdot \boldsymbol{k}^{(h)}_1} & -\infty & -\infty \\[1em]
\textcolor{blue}{\boldsymbol{q}^{(h)}_2 \cdot \boldsymbol{k}^{(h)}_1} & \textcolor{blue}{\boldsymbol{q}^{(h)}_2 \cdot \boldsymbol{k}^{(h)}_2} & -\infty \\[1em]
\boldsymbol{q}^{(h)}_3 \cdot \boldsymbol{k}^{(h)}_1 & \boldsymbol{q}^{(h)}_3 \cdot \boldsymbol{k}^{(h)}_2 & \boldsymbol{q}^{(h)}_3 \cdot \boldsymbol{k}^{(h)}_3
\end{bmatrix}
$$

Attention weights

$$
\boldsymbol{A}^{(h)} =
\begin{bmatrix}
\textcolor{blue}{\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_1\cdot \boldsymbol{k}^{(h)}_1\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_1\cdot \boldsymbol{k}^{(h)}_1\big) }} &
0 & 0 \\[2.5em]
\textcolor{blue}{\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_1\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_2\big) }} &
\textcolor{blue}{\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_2\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_2\cdot \boldsymbol{k}^{(h)}_2\big) }} &
0 \\[2.5em]
\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_1\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_2\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_3\big) } &
\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_2\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_2\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_3\big) } &
\dfrac{ \exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_3\big) }
      { \exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_1\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_2\big)
       +\exp\!\big(\boldsymbol{q}^{(h)}_3\cdot \boldsymbol{k}^{(h)}_3\big) }
\end{bmatrix}
$$

Single head attention output

$$
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{o}^{(h)}_1} \\[1em]
\textcolor{blue}{\boldsymbol{o}^{(h)}_2} \\[1em]
\boldsymbol{o}^{(h)}_3
\end{bmatrix}=
\begin{bmatrix}
\textcolor{blue}{A^{(h)}_{11} \boldsymbol{v}^{(h)}_1} \\[1em]
\textcolor{blue}{A^{(h)}_{21} \boldsymbol{v}^{(h)}_1 +
A^{(h)}_{22} \boldsymbol{v}^{(h)}_2} \\[1em]
A^{(h)}_{31} \boldsymbol{v}^{(h)}_1 +
A^{(h)}_{32} \boldsymbol{v}^{(h)}_2 +
A^{(h)}_{33} \boldsymbol{v}^{(h)}_3
\end{bmatrix}
$$

### Multi head attention output

$$
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{o}_1} \\[1em]
\textcolor{blue}{\boldsymbol{o}_2} \\[1em]
\boldsymbol{o}_3
\end{bmatrix}=
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{o}_1^{(1)}} & \textcolor{blue}{\boldsymbol{o}_1^{(2)}} & \ldots &  \textcolor{blue}{\boldsymbol{o}_1^{(H)}}\\[1em]
\textcolor{blue}{\boldsymbol{o}_2^{(1)}} & \textcolor{blue}{\boldsymbol{o}_2^{(2)}} &  \ldots  &\textcolor{blue}{\boldsymbol{o}_2^{(H)}}\\[1em]
\boldsymbol{o}_3^{(1)} & \boldsymbol{o}_3^{(2)} &  \ldots  &\boldsymbol{o}_3^{(H)}\\
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{W}_O^{(1)} \\[1em]
\boldsymbol{W}_O^{(2)} \\[1em]
\vdots \\
\boldsymbol{W}_O^{(H)} \\
\end{bmatrix}=
\begin{bmatrix}
\textcolor{blue}{\sum_{h=1}^H \boldsymbol{o}_1^{(h)}\boldsymbol{W}_O^{(h)}}  \\[1em]
\textcolor{blue}{\sum_{h=1}^H \boldsymbol{o}_2^{(h)}\boldsymbol{W}_O^{(h)}}\\[1em]
\sum_{h=1}^H \boldsymbol{o}_3^{(h)}\boldsymbol{W}_O^{(h)}\\
\end{bmatrix}
$$

### residual connection

$$
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{x}^{\text{MLP}}_1} \\[1em]
\textcolor{blue}{\boldsymbol{x}^{\text{MLP}}_2} \\[1em]
\boldsymbol{x}^{\text{MLP}}_3
\end{bmatrix}=
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{o}_1 + \boldsymbol{x}_1}\\[1em]
\textcolor{blue}{\boldsymbol{o}_2 + \boldsymbol{x}_2} \\[1em]
\boldsymbol{o}_3 + \boldsymbol{x}_3
\end{bmatrix}
$$

### MLP

$$
\begin{bmatrix}
\textcolor{blue}{\boldsymbol{o}_1^{\text{MLP}}} \\[1em]
\textcolor{blue}{\boldsymbol{o}_2^{\text{MLP}}} \\[1em]
\boldsymbol{o}_3^{\text{MLP}}
\end{bmatrix}=
\begin{bmatrix}
\textcolor{blue}{\text{MLP}\Big(\text{norm}(\boldsymbol{x}^{\text{MLP}}_1) \Big)} \\[1em]
\textcolor{blue}{\text{MLP}\Big(\text{norm}(\boldsymbol{x}^{\text{MLP}}_2) \Big)} \\[1em]
\text{MLP}\Big(\text{norm}(\boldsymbol{x}^{\text{MLP}}_3) \Big) \\
\end{bmatrix}
$$

### read out layer 

$$
\boldsymbol{x}_4 = \text{read out} \left(\text{norm}(\boldsymbol{o}_3^{\text{MLP}} + \boldsymbol{x}^{\text{MLP}}_3) \right)
$$

All the blue parts are **recomputations** we could have avoided: they do not depend on $\boldsymbol{x}_3$ at all, and they will also not change when we later generate $\boldsymbol{x}_5$, $\boldsymbol{x}_6$, and so on.

So a naive caching strategy is to store all those values and reuse them whenever we extend the sequence, instead of recomputing them from scratch each time.

# KV cache

KV cache is exactly this idea, but in a more focused and memory-efficient form.
One thing we haven't considered is this:

- For the purpose of predicting $\boldsymbol{x}_4$, the only thing we need is the *last* token’s representation.

Let’s look more closely at what we actually need to compute for the last token (token 3).
This will make it clear which quantities have to be recomputed and which ones can safely be reused from a cache.

Self-attention block:

We have

$$
\boldsymbol{o}_3^{\text{Attn}} = \sum_{h=1}^H \boldsymbol{o}_3^{(h)}\boldsymbol{W}_O^{(h)}
$$

where the output of head $h$ is

$$
\boldsymbol{o}_3^{(h)} = A^{(h)}_{31} \boldsymbol{v}^{(h)}_1 +
A^{(h)}_{32} \boldsymbol{v}^{(h)}_2 +
A^{(h)}_{33} \boldsymbol{v}^{(h)}_3,
$$

$$
\begin{bmatrix}
A^{(h)}_{31}, & A^{(h)}_{32}, & A^{(h)}_{33}
\end{bmatrix}
=
\text{Softmax}\left( \begin{bmatrix}
\boldsymbol{q}^{(h)}_3 \cdot \boldsymbol{k}^{(h)}_1, & \boldsymbol{q}^{(h)}_3 \cdot \boldsymbol{k}^{(h)}_2, & \boldsymbol{q}^{(h)}_3 \cdot \boldsymbol{k}^{(h)}_3
\end{bmatrix}\right)
$$

So basically we need

$$
\Big\{\boldsymbol{q}^{(h)}_3, \;\boldsymbol{k}^{(h)}_3,\; \boldsymbol{v}^{(h)}_3, \; \boldsymbol{k}^{(h)}_1,\boldsymbol{k}^{(h)}_2,\;
        \boldsymbol{v}^{(h)}_1,\boldsymbol{v}^{(h)}_2\Big\}_{h=1}^H
$$

We can further split this into:

- **What we need to compute** (depends on $\boldsymbol{x}_3$):

$$
\Big\{\boldsymbol{q}^{(h)}_3,\; \boldsymbol{k}^{(h)}_3,\; \boldsymbol{v}^{(h)}_3\Big\}_{h=1}^H,
$$

where

$$
\boldsymbol{q}^{(h)}_3 = \widetilde{\boldsymbol{x}}_3 \boldsymbol{W}^{(h)}_Q, \qquad
\boldsymbol{k}^{(h)}_3 = \widetilde{\boldsymbol{x}}_3 \boldsymbol{W}^{(h)}_K, \qquad
\boldsymbol{v}^{(h)}_3 = \widetilde{\boldsymbol{x}}_3 \boldsymbol{W}^{(h)}_V.
$$

- **What we can reuse**:

$$
\Big\{\boldsymbol{k}^{(h)}_1,\boldsymbol{k}^{(h)}_2,\;
    \boldsymbol{v}^{(h)}_1,\boldsymbol{v}^{(h)}_2\Big\}_{h=1}^H.
$$

Then, the rest of the forward pass depends only on token 3.

Residual:

$$
\boldsymbol{x}^{\text{MLP}}_3 = \boldsymbol{o}_3^{\text{Attn}} + \boldsymbol{x}_3.
$$

MLP:

$$
\boldsymbol{o}^{\text{MLP}}_3 = \text{MLP}\Big(\text{norm}(\boldsymbol{x}^{\text{MLP}}_3) \Big).
$$

Readout:

$$
\boldsymbol{x}_4 = \text{read out} \left(\text{norm}(\boldsymbol{o}_3^{\text{MLP}} + \boldsymbol{x}^{\text{MLP}}_3) \right)
$$

This is exactly the structure that the KV cache exploits: we recompute only what depends on the newest token, and we reuse everything that belongs to the past.

In general, each decoding step works like this.

For each layer $l$ (and each head $h$ inside that layer):

1. **Compute projections for the new token**  
   For the newly appended token at position $t$, we compute $\boldsymbol{q}^{(l,h)}_t, \; \boldsymbol{k}^{(l,h)}_t, \; \boldsymbol{v}^{(l,h)}_t$.

2. **Read all previously cached keys and values**  
   We load $\{\boldsymbol{k}^{(l,h)}_1, \dots, \boldsymbol{k}^{(l,h)}_{t-1}\}$ and $\{\boldsymbol{v}^{(l,h)}_1, \dots, \boldsymbol{v}^{(l,h)}_{t-1}\}$ from the KV cache.  
   Due to causal masking, these past entries never change after they are first computed.

3. **Do attention only for the new token**  
   We compute attention scores using $\boldsymbol{q}^{(l,h)}_t$ against all cached keys, apply softmax, and then compute the single-token output $\boldsymbol{o}^{(l,h)}_t$.

4. **Append the new key and value to the cache**  
   After using $\boldsymbol{k}^{(l,h)}_t$ and $\boldsymbol{v}^{(l,h)}_t$ for attention, we store them so they can be reused in all future steps.

5. **Finish the rest of the layer normally**  
   The attention outputs across heads are combined with the output projection, added to the residual stream, normalized, and passed through the MLP.  
   Importantly, **these computations happen only for token $t$**, we do not recompute anything for previous tokens.

This repeats for every layer from bottom to top, and the final hidden state of token $t$ is used by the readout layer to predict token $t+1$.

That’s it. With the full computation laid out, the motivation for KV cache should now be clear.