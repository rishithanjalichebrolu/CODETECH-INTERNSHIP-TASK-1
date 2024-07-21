#INTERNSHIP TASK-1
#IT GENERATES THE IMAGE BASED ON THE PROMPT GIVEN BY USERS

!pip install diffusers torch safetensors
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import IPython.display as display

def load_model(model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    # Load the pre-trained model
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    pipe.to("cuda")
    return pipe

def generate_image(pipe, prompt):
    # Generate the image
    image = pipe(prompt=prompt).images[0]
    return image

def main():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = load_model(model_id)

    while True:
        prompt = input("Enter a description for the image (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        
        image = generate_image(pipe, prompt)
        
        # Resize the image
        max_size = (512, 512)  # Example size, you can adjust as needed
        image.thumbnail(max_size, Image.ANTIALIAS)
        
        display.display(image)

if __name__ == "__main__":
    main()
