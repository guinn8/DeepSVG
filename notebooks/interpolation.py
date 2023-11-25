import sys
sys.path.append('/home/guinn8/Code')
sys.path.append('/home/guinn8/Code/deepsvg')


import sys
import torch
from deepsvg.svglib.svg import SVG
from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify, linear
from configs.deepsvg.hierarchical_ordered import Config

# Append paths for importing
sys.path.append('/home/guinn8/Code')
sys.path.append('/home/guinn8/Code/deepsvg')

# Configuration and model setup
pretrained_path = "/home/guinn8/Code/deepsvg/pretrained/hierarchical_ordered.pth.tar"
cfg = Config()
model = cfg.make_model().to(torch.device("cpu"))
utils.load_model(pretrained_path, model)
model.eval()
dataset = load_dataset(cfg)

# Utility Functions
def easein_easeout(t):
    return t * t / (2. * (t * t - t) + 1.)

def encode_icon(idx):
    data = dataset.get(id=idx, random_aug=False)
    model_args = batchify((data[key] for key in cfg.model_args), torch.device("cpu"))
    with torch.no_grad():
        return model(*model_args, encode_mode=True)

def decode(z, do_display=True, return_svg=False, return_png=False):
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths().set_color("random")
    return svg_path_sample.draw(do_display=do_display, return_png=return_png) if not return_svg else svg_path_sample

def interpolate(z1, z2, n=25, filename=None, ease=True, do_display=True):
    alphas = torch.linspace(0., 1., n)
    alphas = easein_easeout(alphas) if ease else alphas
    z_list = [(1 - a) * z1 + a * z2 for a in alphas]
    img_list = [decode(z, do_display=False, return_png=True) for z in z_list]
    to_gif(img_list + img_list[::-1], file_path="/home/guinn8/Code/deepsvg/interp.gif", frame_duration=1/12)

# Main Functionality
def interpolate_icons(idx1=None, idx2=None, n=25, *args, **kwargs):
    z1, z2 = encode_icon(idx1), encode_icon(idx2)
    interpolate(z1, z2, n=n, *args, **kwargs)

# Execution
id1, id2 = dataset.random_id(), dataset.random_id()
interpolate_icons(id1, id2)
