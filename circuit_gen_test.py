"""Standalone schematic dataset generator (VS Code / regular Python).

Install once in the Python environment selected in VS Code:
    python -m pip install schemdraw matplotlib pillow numpy
Run:
    python generate_circuits.py
Small preview run:
    python generate_circuits.py --per-class 10 --output circuit_preview

Default output (beside this script):
    generated_circuits/<class_name>/circuit_00000.png
    generated_circuits/labels.csv
    generated_circuits/config.json
    generated_circuits/preview_224.jpg

Default: 500 images per class, 2500 total, 512 x 512 RGB PNGs.
Only schematic images and their dataset metadata/previews are generated.
There is no model loading, training, notebook access, or training-code output.
Train/validation/test partitioning is left for the later training workflow.
The CSV retains layout_id and template_id to support grouped splitting later.
Layouts are related parameterized drawings, not independent circuit topologies.

Labels are upright after orientation changes and placed only in empty regions.
Unplaceable labels are omitted rather than painting over wires. Geometry is
square-padded without stretching. The preview shows drawings at 224 x 224.
DPI controls intermediate rendering; IMAGE_SIZE controls final dimensions.
A nonempty output folder is refused so existing datasets are never overwritten.
Use the same seed/settings/software/fonts to reproduce a run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import schemdraw
import schemdraw.elements as elm

OUTPUT_DIR = Path(__file__).resolve().parent / 'generated_circuits2'
IMAGES_PER_CLASS = 500
SEED = 42
DPI = 180
IMAGE_SIZE = 512
OPAMP_PROBABILITY = 0.35
RANDOM_ORIENTATION = True
COMPLEXITY = 'simple'  # Start here; mixed adds two-stage circuits.
# medium/complex both cap cascades at two stages to stay legible at 224px.
CATEGORIES = ['voltage_divider', 'current_divider', 'low_pass_filter',
              'high_pass_filter', 'band_pass_filter']
# Routing / branch placement templates; all are available during generation.
LAYOUTS = list(range(9))
FAMILIES = {
    'voltage_divider': ['resistive_ladder', 'series_taps'],
    'current_divider': ['parallel_R', 'parallel_series_R'],
    'low_pass_filter': ['RC', 'RL'],
    'high_pass_filter': ['RC', 'RL'],
    'band_pass_filter': ['series_LC_load_R', 'RC_high_then_low'],
}
R_VALUES = [1000, 2200, 3300, 4700, 6800, 10000, 15000, 22000, 47000]
C_VALUES = [10e-9, 22e-9, 47e-9, 100e-9, 220e-9, 470e-9]
L_VALUES = [1e-3, 2.2e-3, 4.7e-3, 10e-3, 22e-3, 47e-3]


def wire(d, a, b):
    if tuple(a) != tuple(b):
        d.add(elm.Line().at(a).to(b))


def dot(d, p):
    d.add(elm.Dot().at(p))


class Circuit:
    def __init__(self, d, rng):
        self.d, self.rng = d, rng
        self.labels = []
        self.parts = []
        self.counts = dict(R=0, C=0, L=0)

    def text(self, point, text):
        self.labels.append((point, text))

    def part(self, kind, a, b, value=None):
        self.counts[kind] += 1
        name = f'{kind}{self.counts[kind]}'
        if value is None:
            value = self.rng.choice({'R': R_VALUES, 'C': C_VALUES, 'L': L_VALUES}[kind])
        symbol = {'R': elm.Resistor, 'C': elm.Capacitor, 'L': elm.Inductor2}[kind]
        self.d.add(symbol().at(a).to(b))
        unit_value = {'R': (value / 1000, 'kΩ'), 'C': (value * 1e9, 'nF'),
                      'L': (value * 1000, 'mH')}[kind]
        self.text(((a[0]+b[0])/2, (a[1]+b[1])/2),
                  f'{name} {unit_value[0]:g} {unit_value[1]}')
        self.parts.append(dict(name=name, kind=kind, value_si=value))

    def output(self, node):
        # Short output stub on the final node.
        end = (node[0]+0.8, node[1])
        dot(self.d, node)
        wire(self.d, node, end)
        self.d.add(elm.Dot(open=True).at(end))
        self.text(end, 'Vo')


def stage_nodes(layout, n, spacing, height):
    """Nine explicit arrangements for circuit nodes."""
    profiles = {
        0: [height, height], 1: [height, height+1.4],
        2: [height+1.4, height], 3: [height, height-1.2],
        4: [height-1.2, height], 5: [height+0.8, height-0.8],
        6: [height-0.8, height+0.8], 7: [height+1.6, height-0.6],
        8: [height-0.6, height+1.6],
    }
    return [(spacing*(i+1), profiles[layout][i]) for i in range(n)]


def connect_to(d, previous, start, layout):
    # A horizontal/vertical dogleg in reserved space left of the component.
    fraction = [0.2, 0.5, 0.8, 0.35, 0.65, 0.3, 0.7, 0.4, 0.6][layout]
    elbow = previous[0] + (start[0]-previous[0])*fraction
    wire(d, previous, (elbow, previous[1]))
    wire(d, (elbow, previous[1]), (elbow, start[1]))
    wire(d, (elbow, start[1]), start)


def buffer(c, node):
    # Standard Opamp in1 = minus, in2 = plus. Align plus with the signal.
    amp = c.d.add(elm.Opamp().anchor('in2').at((node[0]+2, node[1])))
    wire(c.d, node, amp.in2)
    minus, out = amp.in1, amp.out
    # Feedback runs above the symbol, avoiding the non-inverting input wire.
    xleft, xright = minus[0]-0.6, out[0]+0.8
    ytop = minus[1]+1.4
    wire(c.d, minus, (xleft, minus[1]))
    wire(c.d, (xleft, minus[1]), (xleft, ytop))
    wire(c.d, (xleft, ytop), (xright, ytop))
    wire(c.d, (xright, ytop), (xright, out[1]))
    wire(c.d, out, (xright, out[1]))
    dot(c.d, (xright, out[1]))
    c.output((xright, out[1]))


def build(c, category, family, layout, level, active):
    d, rng = c.d, c.rng
    height, spacing = rng.uniform(4.2, 5.0), rng.uniform(5.3, 6.2)
    n = 1 if level == 'simple' else 2
    # Both dividers get both source types; filters use a sinusoidal voltage input.
    source_kind = rng.choice(['voltage', 'current']) if 'divider' in category else 'ac_voltage'
    symbol = {'voltage': elm.SourceV, 'current': elm.SourceI,
              'ac_voltage': elm.SourceSin}[source_kind]
    d.add(symbol().at((0, 0)).to((0, height)))
    d.add(elm.Ground().at((0, 0)))
    c.text((0, height/2), 'S1')  # neutral identifier, never a class title
    previous = (0, height)
    nodes = stage_nodes(layout, n, spacing, height)

    if category == 'current_divider':
        branches = 2 if level == 'simple' else rng.choice([3, 4])
        # Parallel branches share both buses; route the upper bus as a staircase.
        nodes = stage_nodes(layout, 2, spacing, height)
        nodes = [(spacing*(i+1), nodes[i % 2][1]) for i in range(branches)]
        for i, node in enumerate(nodes):
            connect_to(d, previous, node, layout)
            bottom = (node[0], 0)
            if family == 'parallel_series_R':
                mid = (node[0], node[1]/2)
                c.part('R', node, mid)
                c.part('R', mid, bottom)
            else:
                c.part('R', node, bottom)
            dot(d, node)
            dot(d, bottom)
            previous = node
        wire(d, (0, 0), (nodes[-1][0], 0))
        # Use the same voltage test point as other classes, avoiding an Io-only
        # label/arrow shortcut. The parallel resistors still divide input current.
        c.output(nodes[-1])
        return source_kind

    if category == 'voltage_divider' and family == 'series_taps':
        # Two series resistors, with an intermediate output tap. The second
        # resistor is routed on a separate column to vary the string layout.
        first, last = stage_nodes(layout, 2, spacing, height)
        start = (first[0]-3.3, first[1])
        connect_to(d, previous, start, layout)
        c.part('R', start, first)
        connect_to(d, first, last, layout)
        c.part('R', last, (last[0], 0))
        wire(d, (0, 0), (last[0], 0))
        # Mark output on the last node, away from intermediate routing.
        c.output(last)
        return source_kind

    if category == 'band_pass_filter' and family == 'RC_high_then_low':
        # High-pass C/R followed by low-pass R/C. R2 >> R1 limits loading;
        # nominal low-pass corner is 10x the high-pass corner.
        nodes = stage_nodes(layout, 2, spacing, height)
        r1 = rng.choice([1000, 2200, 4700])
        c1 = rng.choice(C_VALUES)
        r2, c2 = 10*r1, c1/100
        sections = [('C', 'R', c1, r1), ('R', 'C', r2, c2)]
    else:
        mapping = {
            ('voltage_divider', 'resistive_ladder'): ('R', 'R'),
            ('low_pass_filter', 'RC'): ('R', 'C'),
            ('low_pass_filter', 'RL'): ('L', 'R'),
            ('high_pass_filter', 'RC'): ('C', 'R'),
            ('high_pass_filter', 'RL'): ('R', 'L'),
            ('band_pass_filter', 'series_LC_load_R'): ('LC', 'R'),
        }
        series, shunt = mapping[(category, family)]
        sections = [(series, shunt, None, None)]*len(nodes)

    for node, (series, shunt, sv, pv) in zip(nodes, sections):
        start = (node[0]-3.3, node[1])
        connect_to(d, previous, start, layout)
        if series == 'LC':
            mid = ((start[0]+node[0])/2, node[1])
            kinds = rng.choice([('L', 'C'), ('C', 'L')])
            c.part(kinds[0], start, mid)
            c.part(kinds[1], mid, node)
        else:
            c.part(series, start, node, sv)
        c.part(shunt, node, (node[0], 0), pv)
        dot(d, node)
        dot(d, (node[0], 0))
        previous = node
    wire(d, (0, 0), (previous[0], 0))
    if active:
        buffer(c, previous)
    else:
        c.output(previous)
    return source_kind


def get_font(size):
    for name in ['DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(c, path, size, dpi):
    wrapper = c.d.draw(show=False)
    fig, ax = wrapper.fig, wrapper.ax
    fig.set_dpi(dpi)
    fig.canvas.draw()
    raw = io.BytesIO()
    fig.savefig(raw, dpi=dpi, facecolor='white')
    raw.seek(0)
    with Image.open(raw) as im:
        base = im.convert('RGB')
    w, h = base.size
    labels = []
    for p, msg in c.labels:
        x, y = ax.transData.transform(p)
        labels.append((float(x), float(h-y), msg))
    angle = c.rng.choice([0, 90, 180, 270]) if RANDOM_ORIENTATION else 0
    mirrored = c.rng.choice([False, True]) if RANDOM_ORIENTATION else False
    if mirrored:
        base = ImageOps.mirror(base)
        labels = [(w-x, y, msg) for x, y, msg in labels]
    if angle == 90:
        labels = [(y, w-x, msg) for x, y, msg in labels]
    elif angle == 180:
        labels = [(w-x, h-y, msg) for x, y, msg in labels]
    elif angle == 270:
        labels = [(h-y, x, msg) for x, y, msg in labels]
    if angle:
        base = base.rotate(angle, expand=True, fillcolor='white')
    # Render first at actual DPI, scale geometry to a square working area, and
    # paint text upright at a size that remains useful after 224px downsampling.
    margin = 44
    scale = min((size-2*margin)/base.width, (size-2*margin)/base.height)
    nw, nh = max(1, round(base.width*scale)), max(1, round(base.height*scale))
    ox, oy = (size-nw)//2, (size-nh)//2
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(base.resize((nw, nh), Image.Resampling.LANCZOS), (ox, oy))
    occupied = np.asarray(canvas.convert('L')) < 235
    draw = ImageDraw.Draw(canvas)
    font = get_font(max(10, round(size/36)))
    omitted = 0
    for x, y, msg in labels:
        # Independently omit some values in every class to reduce text reliance.
        if c.rng.random() < 0.15:
            continue
        cx, cy = x*nw/w + ox, y*nh/h + oy
        # Rotation swaps geometry dimensions, so use transformed source sizes.
        if angle in [90, 270]:
            cx, cy = x*nw/h + ox, y*nh/w + oy
        bbox = draw.textbbox((0, 0), msg, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        found = None
        for radius in [18, 26, 36, 48, 60]:
            for theta in [-math.pi/2, math.pi/2, 0, math.pi,
                          -math.pi/4, math.pi/4, 3*math.pi/4, -3*math.pi/4]:
                left = round(cx+radius*math.cos(theta)-tw/2)
                top = round(cy+radius*math.sin(theta)-th/2)
                rect = (left-3, top-3, left+tw+3, top+th+3)
                a, b, e, f = rect
                if a < 3 or b < 3 or e > size-3 or f > size-3:
                    continue
                if not occupied[b:f, a:e].any():
                    found = (left, top, rect)
                    break
            if found:
                break
        if found is None:
            omitted += 1
            continue
        left, top, (a, b, e, f) = found
        # Never erase the schematic; accept only completely clear rectangles.
        draw.text((left-bbox[0], top-bbox[1]), msg, fill='black', font=font)
        occupied[b:f, a:e] = True
    canvas.save(path, dpi=(dpi, dpi))
    return angle, mirrored, omitted


def generate(output=OUTPUT_DIR, per_class=IMAGES_PER_CLASS, seed=SEED,
             size=IMAGE_SIZE, dpi=DPI):
    """Generate class-organized images and a metadata CSV; no training steps."""
    output = Path(output).expanduser().resolve()
    if not isinstance(per_class, int) or per_class < 1:
        raise ValueError('Use at least one image per class.')
    if size < 224 or dpi < 50:
        raise ValueError('Use size >= 224 and DPI >= 50.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f'{output} is not empty. Choose a fresh folder with --output.'
        )
    if not 0 <= OPAMP_PROBABILITY <= 1:
        raise ValueError('OPAMP_PROBABILITY must be between zero and one.')
    if COMPLEXITY not in ['mixed', 'simple', 'medium', 'complex']:
        raise ValueError('COMPLEXITY must be mixed, simple, medium or complex.')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        generator_version='9-standalone', seed=seed, images_per_class=per_class,
        total_images=per_class*len(CATEGORIES), image_size=size, dpi=dpi,
        categories=CATEGORIES, layouts=LAYOUTS, families=FAMILIES,
        schemdraw_version=schemdraw.__version__, complexity=COMPLEXITY,
        opamp_probability=OPAMP_PROBABILITY, random_orientation=RANDOM_ORIENTATION,
    )
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    columns = [
        'filename', 'circuit_type', 'family', 'layout_id', 'template_id',
        'complexity', 'seed', 'source_type', 'uses_opamp',
        'rotation_degrees_ccw', 'mirrored', 'omitted_collision_labels',
        'resistor_count', 'capacitor_count', 'inductor_count', 'parts_json', 'sha256',
    ]
    hashes = set()
    previews = []
    with (output/'labels.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for ci, category in enumerate(CATEGORIES):
            folder = output/category
            folder.mkdir()
            for i in range(per_class):
                # Stable per-image seeds without reliance on Python's hash().
                seed_text = f'{seed}:{category}:{i}'.encode('utf-8')
                sample_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], 'big')
                rng = random.Random(sample_seed)
                family = FAMILIES[category][i % len(FAMILIES[category])]
                layout = LAYOUTS[(i//len(FAMILIES[category])) % len(LAYOUTS)]
                level = rng.choice(['simple', 'medium']) if COMPLEXITY == 'mixed' else COMPLEXITY
                active = 'filter' in category and rng.random() < OPAMP_PROBABILITY
                path = folder/f'circuit_{i:05d}.png'
                d = schemdraw.Drawing(show=False)
                d.config(unit=3, color='black', lw=rng.uniform(1.6, 2.4))
                c = Circuit(d, rng)
                try:
                    source_kind = build(c, category, family, layout, level, active)
                    angle, mirrored, omitted = render(c, path, size, dpi)
                finally:
                    plt.close('all')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hashes:
                    raise RuntimeError('Duplicate image generated; use another seed in a fresh folder.')
                hashes.add(digest)
                writer.writerow(dict(
                    filename=path.relative_to(output).as_posix(), circuit_type=category,
                    family=family, layout_id=layout,
                    template_id=f'{category}/{family}/layout_{layout}', complexity=level,
                    seed=sample_seed, source_type=source_kind, uses_opamp=active,
                    rotation_degrees_ccw=angle, mirrored=mirrored,
                    omitted_collision_labels=omitted, resistor_count=c.counts['R'],
                    capacitor_count=c.counts['C'], inductor_count=c.counts['L'],
                    parts_json=json.dumps(c.parts), sha256=digest,
                ))
                if i < 3:
                    previews.append((path, f'{category} / {family}', ci, i))
                if (i+1) % 25 == 0 or i+1 == per_class:
                    f.flush()
                    print(f'{category}: {i+1}/{per_class}', flush=True)
    sheet = Image.new('RGB', (3*224, len(CATEGORIES)*260), 'white')
    painter = ImageDraw.Draw(sheet)
    for path, title, row, col in previews:
        x, y = col*224, row*260
        with Image.open(path) as im:
            sheet.paste(im.resize((224, 224), Image.Resampling.BILINEAR), (x, y))
        category, family = title.split(' / ')
        painter.text((x+4, y+225), category, fill='black', font=get_font(11))
        painter.text((x+4, y+240), family, fill='black', font=get_font(11))
    sheet.save(output/'preview_224.jpg')
    print(f'\nGenerated {per_class*len(CATEGORIES)} images in {output}')
    print(f'Labels CSV: {output / "labels.csv"}')
    print(f'Preview: {output / "preview_224.jpg"}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--per-class', type=int, default=IMAGES_PER_CLASS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--dpi', type=int, default=DPI)
    args = parser.parse_args()
    generate(args.output, args.per_class, args.seed, args.size, args.dpi)
