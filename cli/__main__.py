import torch

from imageclassifier.image import get_image
from imageclassifier.model import predict

model = torch.load("models/model.pth")

animals_file = open("cli/animals.txt", "r")
animals_list = animals_file.read().splitlines()
animals_file.close()

try:
    image_location = input("Enter the URL or local path to an image: ")
    image = get_image(image_location)
except:
    print("Invalid URL or path.")
    exit()

animal, probability = predict(model, image)
print("Predicted animal:", animals_list[animal])
print("Confidence:", round(probability, 5))