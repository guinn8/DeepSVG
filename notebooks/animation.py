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


class SVGAnimator:
    def __init__(self):
        self.pretrained_path = "/home/guinn8/Code/deepsvg/pretrained/hierarchical_ordered.pth.tar"
        self.cfg = Config()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.cfg.make_model().to(self.device)
        utils.load_model(self.pretrained_path, self.model)
        self.model.eval()
        self.dataset = load_dataset(self.cfg)

    def load_svg(self, filename):
        svg = SVG.load_svg(filename)
        svg = self.dataset.simplify(svg)
        svg = self.dataset.preprocess(svg, mean=True)
        return svg

    def easein_easeout(self, t):
        return t * t / (2. * (t * t - t) + 1.)

    def interpolate(self, z1, z2, n=25, filename=None, ease=True, do_display=True):
        alphas = torch.linspace(0., 1., n)
        if ease:
            alphas = self.easein_easeout(alphas)
        z_list = [(1 - a) * z1 + a * z2 for a in alphas]
        img_list = [self.decode(z, do_display=False, return_png=True) for z in z_list]
        to_gif(img_list + img_list[::-1], file_path=filename, frame_duration=1 / 12)

    def encode(self, data):
        model_args = batchify((data[key] for key in self.cfg.model_args), self.device)
        with torch.no_grad():
            return self.model(*model_args, encode_mode=True)

    def encode_icon(self, idx):
        data = self.dataset.get(id=idx, random_aug=False)
        return self.encode(data)

    def encode_svg(self, svg):
        data = self.dataset.get(svg=svg)
        return self.encode(data)

    def decode(self, z, do_display=True, return_svg=False, return_png=False):
        commands_y, args_y = self.model.greedy_sample(z=z)
        tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
        svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths().set_color("random")
        return svg_path_sample.draw(do_display=do_display, return_png=return_png) if not return_svg else svg_path_sample

    def open_file_with_default_app(self, file_path):
        try:
            subprocess.run(["xdg-open", file_path], check=True)
        except Exception as e:
            print(f"Error opening file: {e}")

    def interpolate_icons(self, idx1=None, idx2=None, n=25, *args, **kwargs):
        z1, z2 = self.encode_icon(idx1), self.encode_icon(idx2)
        self.interpolate(z1, z2, n=n, *args, **kwargs)


def main():
    animator = SVGAnimator()
    id1, id2 = animator.dataset.random_id(), animator.dataset.random_id()
    output_file = "/home/guinn8/Code/deepsvg/output.gif"
    animator.interpolate_icons(id1, id2, filename=output_file)
    animator.open_file_with_default_app(output_file)


if __name__ == "__main__":
    main()