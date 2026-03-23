from diffusers import StableDiffusionPipeline
import torch
import os
import csv

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda" 
   if torch.cuda.is_available() else "cpu")

pipe.enable_attention_slicing()

weather = ["sunny", "rainy", "foggy", "night"]
scene = ["urban street", "highway", "intersection"]
traffic = ["low traffic", "moderate traffic", "heavy traffic"]

os.makedirs("dataset", exist_ok=True)

with open("dataset/labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "weather", "scene", "traffic"])

    count = 0

    for w in weather:
        for s in scene:
            for t in traffic:

                prompt = f"{s}, {w}, {t}, cars, realistic driving scene"

                print(f"Generating: {prompt}")

                image = pipe(prompt).images[0]
                filename = f"img_{count}.png"
                image.save(f"dataset/{filename}")

                writer.writerow([filename, w, s, t])
                count += 1

print("Dataset generation complete!")