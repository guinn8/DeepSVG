"""
Complete Python script extracted from 'animation.ipynb', 
including all necessary functionalities and configurations.
"""

import sys
sys.path.append('/home/guinn8/Code')
sys.path.append('/home/guinn8/Code/deepsvg')
import torch
from deepsvg.svglib.svg import SVG
from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify, linear
from configs.deepsvg.hierarchical_ordered import Config
import subprocess
import os

# Configuration and model setup
pretrained_path = "/home/guinn8/Code/deepsvg/pretrained/hierarchical_ordered.pth.tar"
cfg = Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define the device here
model = cfg.make_model().to(device)
utils.load_model(pretrained_path, model)
model.eval()
dataset = load_dataset(cfg)

def load_svg(filename):
    svg = SVG.load_svg(filename)
    svg = dataset.simplify(svg)
    svg = dataset.preprocess(svg, mean=True)
    return svg

def easein_easeout(t):
    return t*t / (2. * (t*t - t) + 1.);

def interpolate(z1, z2, n=25, filename=None, ease=True, do_display=True):
    alphas = torch.linspace(0., 1., n)
    if ease:
        alphas = easein_easeout(alphas)
    z_list = [(1-a) * z1 + a * z2 for a in alphas]
    
    img_list = [decode(z, do_display=False, return_png=True) for z in z_list]
    to_gif(img_list + img_list[::-1], file_path=filename, frame_duration=1/12)

def encode(data):
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z

def encode_icon(idx):
    data = dataset.get(id=idx, random_aug=False)
    return encode(data)
    
def encode_svg(svg):
    data = dataset.get(svg=svg)
    return encode(data)

def decode(z, do_display=True, return_svg=False, return_png=False):
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths().set_color("random")
    
    if return_svg:
        return svg_path_sample
    
    return svg_path_sample.draw(do_display=do_display, return_png=return_png)

def open_file_with_default_app(file_path):
    try:
        subprocess.run(["xdg-open", file_path], check=True)
    except Exception as e:
        print(f"Error opening file: {e}")

# Main Functionality
def interpolate_icons(idx1=None, idx2=None, n=25, *args, **kwargs):
    z1, z2 = encode_icon(idx1), encode_icon(idx2)
    interpolate(z1, z2, n=n, *args, **kwargs)

# Execution Logic (Example usage)
id1, id2 = dataset.random_id(), dataset.random_id()
output_file = "/home/guinn8/Code/deepsvg/output/output.gif"  # Output file path
interpolate_icons(id1, id2, filename=output_file)

# Open the output file with the default application
open_file_with_default_app(output_file)
