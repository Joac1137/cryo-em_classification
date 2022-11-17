# Data
- Labels
    - Gaussian highlight
    - All white labels
    - Asger special
- Label size
- Zero centered data
- Split images

# Model
- Learning Rate
- Filters (Perhaps preprocessing)
    - Denoice
    - Edge detection
- Comparison
    - Compare Pretrained Models
        - Resnet
        - Imagenet
        - Unet
    - Homemade
        - Basic architecture
        - Unet architecture
- Layers
    - What to do instead of max pooling?
        - Perform convolution with stride = 2.
        - Has the same effect of down-sampling by a factor of two (like max pooling normally does), but information is not thrown away in the same way as with max pooling.
- Dice loss / Intersection-over-Union
- See for instance slides from lecture 6 about hyperparameter tuning.
- Eacly have larger filters and then reduce the size in later layers
- Maybe apply 1x1 convolution in order to do dimension reduction
- Apply skip connection in order to fix vanishing gradiant problem
- Use add instead of concatenate
- Might change accuracy to val_loss in callbacks - might be a very good idea
- Use transpose convolution for up-sampling
- Plot segmentation on top of the original image?

# Experiments
- Large unet model
    - Dice loss vs mse
- Compare models
    - Large unet model
    - basic model
    - large residual unet model