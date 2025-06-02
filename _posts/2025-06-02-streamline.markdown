---
layout: post
title:  "Streamlining Prediction in Bayesian Deep Learning"
date:   2025-04-24 22:21:59 +00:00
image: images/suq.png
categories: research
author: "Rui Li"
authors: "<strong>Rui Li</strong>, Marcus Klasson, Arno Solin, Martin Trapp"
venue: "ICLR"
paper: https://arxiv.org/abs/2411.18425
code: https://github.com/AaltoML/SUQ
---
While estimating posterior has been actively researched in Bayesian deep learning (BDL), how to make predictions with posterior efficiently is largely overlooked. We examine streamlining prediction in BDL through a single forward pass without sampling. We showcase our approach for both MLP and transformers, such as ViT and GPT-2.