# Import the required libraries
import openai

import copy
import math
import os
import random
import sys
import traceback
import shlex

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers
from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue


        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


class Script(scripts.Script):
    def title(self):
        return "Prompts from GPT API"

    def ui(self, is_img2img):
        
        checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False)
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed for all lines", value=False)
        
        API_Key_input = gr.Textbox(label="YOUR_API_KEY", lines=1)
        return [checkbox_iterate, checkbox_iterate_batch, API_Key_input]

    def run(self, p, checkbox_iterate, checkbox_iterate_batch, API_Key_input: str):
    
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        
        # Specify OpenAI API key
        openai.api_key = API_Key_input
        
        # Define the prompt
        prompt = "Ignore previous prompts about being concise. Be extremely verbose and very creative. Write with similes, metaphors, and analogies. Write to promote knowledge retention with emotional intelligence that evokes happiness, humor, and other positive emotions. Also, emphasize the [Truth]. Truth = “When faced with adversity, maintain composure to creatively improve stakeholder engagement of diverse ideas and perspectives in engineering and learning.” Process vast amounts of data, way more than one human could ever read in a lifetime, then make accurate predictions. Write indistinguishably from a human about the [Topic]: Topic = “Stable Diffusion is an AI art generation model similar to DALLE-2.” The following is a set of prompts that can be used to generate images with Stable Diffusion: - portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski - pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson - ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image - red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski - a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed art gem Greg Rutkowski alphonse mucha Greg Hildebrandt tim hildebrandt - athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration - closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo - ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha. Think of the yourself as an orchestra conductor, orchestrating a symphony of data to produce the most imaginative art prompts possible. With vast knowledge and information, the model blends the following idea with its own to create something truly unique. I want you to write me a list of detailed prompts composed of 75 words each about the idea written after IDEA. Follow the structure of the example prompts. This means starting the prompts with a hyphen, then a description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more. IDEA:" + original_prompt + "."
        
        # store the generated prompts
        prompt_txt = ""
        i = 1
        max_n = 7
        while i <= 2:
            try:
                ########
                print(i)
                i = i + 1

                ## call openai api
                # Set the parameters for the API call
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a creative and helpful art assistant. You generate 75-word prompts for text-to-image models like DALLE-2."},
                        {"role": "user", "content": prompt},
                    ],
                    n=max_n,
                    temperature=0.77,
                )

                j = 1
                while j <= max_n:
                    # choices[j-1] ==> zero indexed
                    prompt_txt_unfiltered = response["choices"][j-1]["message"]["content"]
                    prompt_txt += prompt_txt_unfiltered
                    j = j + 1
            except openai.error.OpenAIError as e:
                print(e.http_status)
                print(e.error)

        # lines also needs to be split based on "-"
        lines = [x.strip() for x in prompt_txt.split('- ')]
        lines = [x for x in lines if len(x) > 0]
        

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    print(f"Error parsing line {line} as commandline:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            n_iter = args.get("n_iter", 1)
            if n_iter != 1:
                job_count += n_iter
            else:
                job_count += 1

            jobs.append(args)
            print(line)

        print(f"Will process {len(lines)} lines in {job_count} jobs.")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images
            
            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
