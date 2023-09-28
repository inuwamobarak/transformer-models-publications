# google-efficientnet
# Google EfficientNet for Image Classification

Predicting image classes can now be achieved without extensive training, thanks to the advancements in transformer-based models.

### What are Transformer Models?

Introduced in the groundbreaking "Attention is All You Need" paper by Vaswani et al., transformers leverage attention mechanisms to capture complex patterns and dependencies in sequential data. Initially designed for NLP tasks, the success of transformers has inspired their application to other domains, including computer vision.

![transformer_architecture](https://github.com/inuwamobarak/google-efficientnet/assets/65142149/0d0cf449-0a58-4a69-9244-0bdc1fc1a078)


In the context of image classification, transformers leverage self-attention mechanisms to process images as sequences of patches, breaking down the image into manageable pieces. This approach allows the model to focus on relevant regions and relationships between patches, enabling it to capture intricate spatial patterns effectively.

#### Pre-trained EfficientNet Models

As with most transformer models, transfer learning allows us to leverage the power of EfficientNet without starting from scratch. Transfer learning involves using pre-trained models that have been trained on large-scale datasets. Google and Huggingface offer pre-trained versions of EfficientNet, which can be fine-tuned on specific image classification tasks even with relatively small datasets.


![Screenshot from 2023-07-26 09-31-05](https://github.com/inuwamobarak/google-efficientnet/assets/65142149/dd71e7e6-157c-4506-b563-4d2c557e498e)

### Image Classification using EfficientNet

#### Installing the Transformer Model

Since Google Colaboratory does not have the Transformer library pre-installed, we need to install it first:

```python
!pip install -q datasets transformers
```

#### Loading the EfficientNet Transformer

We load the pre-trained model from Huggingface's model hub:

```python
image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")
inputs = image_processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
```

The output will be the predicted class label for the image.

### Reference Links
- [Google AI Blog - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://ai.googleblog.com/2020/04/efficientnet-improving-accuracy-and.html)
- [Huggingface Transformers Documentation](https://huggingface.co/transformers/)
- ["Attention is All You Need" - Vas
