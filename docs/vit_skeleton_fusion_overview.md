# ViT + Skeleton Transformer Fusion Overview

This repository extends MMAction2 with a lightweight pipeline that fuses a ViT (or any vision backbone) with a transformer that encodes skeleton keypoints. The key additions are summarized below so you can understand and configure the components quickly.

## SkeletonTransformer backbone
- Accepts skeleton tensors shaped `(B, num_clips, num_person, clip_len, num_joints, num_feats)` and flattens the temporal and joint dimensions into tokens.
- Projects joint features to `embed_dims`, adds learned temporal and joint positional embeddings, and encodes the sequence with stacked `nn.TransformerEncoder` layers before mean pooling across tokens and people/clips to produce a single feature vector per sample.
- Parameterized by joint count, feature dimension, depth, heads, MLP ratio, dropout, and positional length so it can be tuned for different skeleton datasets.

## TransformerFusionHead
- Projects the vision and skeleton features into a common hidden space and fuses them by concatenation (default) or element-wise addition.
- Applies dropout and a linear classifier on the fused representation; weight initialization uses truncated normal for stable starts.
- Accepts optional `hidden_channels` to decouple projection size from the incoming feature dimensions.

## RecognizerViTSkeleton
- Wraps a vision backbone, a skeleton backbone, and a fusion head into a two-stream recognizer.
- Uses a `MultiModalDataPreprocessor` by default so RGB and skeleton inputs are normalized and collated together without extra config.
- Pools arbitrary backbone outputs (tensors, lists, or tuples) to vectors, passes them through an optional neck, and forwards them to the fusion head for loss or prediction.

## Tests
- **Backbone**: Confirms the SkeletonTransformer produces finite outputs with the expected `(batch, embed_dims)` shape.
- **Head**: Covers both concat and add fusion modes, verifying logits shape and classification loss computation.
- **Integration**: Builds a RecognizerViTSkeleton with dummy vision/skeleton backbones, exercises `train_step` and `test_step`, and checks predictions are valid probabilities.

Use these components by registering `SkeletonTransformer` as the skeleton backbone and `TransformerFusionHead` as the classifier when configuring a new recognizer. The defaults make it easy to plug into existing ViT configs while adding skeleton awareness.
