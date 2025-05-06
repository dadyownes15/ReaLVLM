
# Models Qwen/Qwen2.5-VL-3B-Instruct
# Create an interface for the different models so it is easy to switch between different models

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



class FoundationalModel:
    def __init__(self):
        # Identification prompt text -> Used for sending the prompt with the video
        # Model name: 
        self.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.identification_text = "You are a helpful assistant that can answer questions about the video."
        self.target_text = ""

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    def set_target_text(self, text):
        self.target_text = f"""ask: Evaluate whether the given image includes {text} on a scale from 0 to 1. A score of 1 means {text} is clearly present in the image, while a score of 0 means {text} is not present at all. For intermediate cases, assign a value between 0 and 1 based on the degree to which {text} is visible.

                            Consideration: The key is whether {text} is present in the image, not its focus. Thus, if {text} is present, even if it is not the main focus, assign a higher score like 1.0.

                            Output: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in one short sentence."""



        
    def inference(self, video_path, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": self.target_text},
                    {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        
        fps_inputs = video_kwargs['fps']
        print("video input:", video_inputs[0].shape)
        num_frames, _, resized_height, resized_width = video_inputs[0].shape
        print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
