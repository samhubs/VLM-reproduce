# Coding PalIGemma from scratch with Umar Jamil
## Steps
1. Define the configuration class `SiglipVisionConfig` containing al the attributes for the transformer. The class `SigLipVisionConfig` is likely used to configure parameters for a vision model. 

2. Create the following classes in a sequence:
-  `SiglipVisionModel(nn.Module)` class and define the `forward` method that takes in a batch of images in the shape `[B, C, H, W]` and turns that into `[B, num_patches, embed_dim]`

- `SiglipVisionTransformer(nn.Module)` where the forward method creates initial and contextualized embeddings

- `SiglipVisionEmbeddings(nn.Module)` that creates embeddings along with the positional encodings