from torchvision.models import ResNet50_Weights

# Get the class mappings from the ResNet50 weights
weights = ResNet50_Weights.DEFAULT
class_mapping = weights.meta["categories"]

# Write the class names to a file
with open("imagenet_classes.txt", "w") as f:
    for class_name in class_mapping:
        f.write(f"{class_name}\n")

print("ImageNet classes file generated successfully.")