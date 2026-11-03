"""V7.7: connected schematic templates + upright collision-aware labels.
Install: pip install schemdraw matplotlib pillow
"""
import csv
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
import schemdraw
import schemdraw.elements as elm

OUTPUT_DIR = Path(r'C:\Users\admin\Documents\generated_circuits')
NUM_IMAGES = 500
OPAMP_PROBABILITY = 0.35  # Chance of a buffered filter version
DPI = 180
RANDOM_ORIENTATION = True
# 'mixed', 'simple', 'medium', or 'complex'
COMPLEXITY = 'mixed'

CATEGORIES = [
    'voltage_divider', 'current_divider', 'low_pass_filter',
    'high_pass_filter', 'band_pass_filter'
]
RESISTORS = [1000, 2200, 3300, 4700, 5000, 6800, 10000, 15000, 22000, 47000]
CAPACITORS = [10, 22, 47, 100, 220, 470]  # nF
INDUCTORS = [1, 2.2, 4.7, 10, 22, 47]  # mH
VOLTAGES = [3, 5, 9, 10, 12, 15, 24]
CURRENTS = [1, 2, 5, 10, 20]  # mA


def line(d, a, b):
    if a != b:
        d.add(elm.Line().at(a).to(b))


def dot(d, p):
    d.add(elm.Dot().at(p))


# Labels are stored in circuit coordinates and painted AFTER image transforms.
LABELS = []


def label(d, p, message):
    LABELS.append((p, message))


def font_for(size):
    paths = [
        r'C:\Windows\Fonts\arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()



class Parts:
    def __init__(self):
        self.counts = {'R': 0, 'C': 0, 'L': 0}

    def add(self, d, kind, a, b):
        self.counts[kind] += 1
        name = f'{kind}{self.counts[kind]}'
        if kind == 'R':
            value = random.choice(RESISTORS)
            part = elm.Resistor()
            value_text = f'{value / 1000:g} kΩ'
        elif kind == 'C':
            value = random.choice(CAPACITORS)
            part = elm.Capacitor()
            value_text = f'{value:g} nF'
        else:
            value = random.choice(INDUCTORS)
            part = elm.Inductor2()
            value_text = f'{value:g} mH'
        # Draw only the component; the label is added upright later.
        d.add(part.at(a).to(b))
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        # Store the component CENTER. Placement is decided AFTER rotation.
        label(d, (mx, my), f'{name} = {value_text}')


def source(d, kind='voltage'):
    """Both source terminals are exactly (0,4) and (0,0)."""
    if kind == 'voltage':
        value = random.choice(VOLTAGES)
        d.add(elm.SourceV().at((0, 0)).to((0, 4)))
        label(d, (0, 2), f'Vin = {value} V')
    else:
        value = random.choice(CURRENTS)
        d.add(elm.SourceI().at((0, 0)).to((0, 4)))
        label(d, (0, 2), f'Iin = {value} mA')
    d.add(elm.Ground().at((0, 0)))


def output(d, node):
    """Output is a short, connected test-point stub."""
    x, y = node
    dot(d, node)
    line(d, node, (x + 0.7, y))
    d.add(elm.Dot(open=True).at((x + 0.7, y)))
    label(d, (x + 0.7, y), 'Vout')


def stage_count(level):
    if level == 'simple':
        return random.randint(1, 2)
    if level == 'medium':
        return random.randint(2, 3)
    return random.randint(3, 4)


def make_voltage_divider(d, p, level):
    """One or more resistor stages, all tapped above the final shunt R."""
    source(d)
    n = stage_count(level)
    x0, top, bottom = 4.5, 4.0, 0.0
    # Resistive ladder: each stage has a series R and a shunt R.
    # Every stage is electrically connected to the source and ground.
    previous = (0, top)
    for i in range(n):
        x = x0 + i * 4.5
        left = (x - 2.5, top)
        right = (x, top)
        line(d, previous, left)
        p.add(d, 'R', left, right)
        p.add(d, 'R', right, (x, bottom))
        dot(d, right)
        previous = right
    line(d, (0, 0), (previous[0], 0))
    output(d, previous)
    return f'resistor_ladder_{n}_stage'


def make_current_divider(d, p, level):
    source(d, 'current')
    count = {'simple': (2, 3), 'medium': (3, 4), 'complex': (4, 5)}[level]
    branches = random.randint(*count)
    first_x = 4.0
    xs = [first_x + i * 4.0 for i in range(branches)]
    line(d, (0, 4), (xs[-1], 4))
    line(d, (0, 0), (xs[-1], 0))
    for x in xs:
        p.add(d, 'R', (x, 4), (x, 0))
        dot(d, (x, 4))
        dot(d, (x, 0))
    output(d, (xs[-1], 4))
    return f'{branches}_parallel_branches'


def add_opamp_buffer(d, input_node):
    """Unity-gain non-inverting op-amp buffer after a passive filter.

    The positive input is driven by the filter; the negative input is
    fed directly from the output, creating genuine negative feedback.
    """
    x, y = input_node
    amp = d.add(elm.Opamp().at((x + 4.2, y - 2.0)))
    # Schemdraw Opamp default anchors: in2 = non-inverting (+),
    # in1 = inverting (-), out = output. Never swap these.
    plus = amp.in2
    minus = amp.in1
    out = amp.out
    if abs(float(plus[1]) - float(minus[1])) < 1e-6:
        raise RuntimeError("Op-amp input anchors overlap; check Schemdraw version")
    # Use the element's actual anchors so wires meet the op-amp pins.
    line(d, input_node, (plus[0] - 0.7, y))
    line(d, (plus[0] - 0.7, y), (plus[0] - 0.7, plus[1]))
    line(d, (plus[0] - 0.7, plus[1]), plus)
    feedback_y = max(y + 2.0, minus[1] + 2.5)
    feedback_x = out[0] + 1.3
    line(d, minus, (minus[0] - 0.45, minus[1]))
    line(d, (minus[0] - 0.45, minus[1]), (minus[0] - 0.45, feedback_y))
    line(d, (minus[0] - 0.45, feedback_y), (feedback_x, feedback_y))
    line(d, (feedback_x, feedback_y), (feedback_x, out[1]))
    line(d, out, (feedback_x, out[1]))
    dot(d, (feedback_x, out[1]))
    output(d, (feedback_x, out[1]))
    label(d, (float(out[0]), float(out[1]) - 1.4), 'U1 (voltage follower)')


def make_low_pass(d, p, level, active=False):
    source(d)
    n = stage_count(level)
    previous = (0, 4)
    for i in range(n):
        x = 4.5 + 4.5 * i
        start = (x - 2.5, 4)
        node = (x, 4)
        line(d, previous, start)
        p.add(d, 'R', start, node)
        p.add(d, 'C', node, (x, 0))
        dot(d, node)
        previous = node
    line(d, (0, 0), (previous[0], 0))
    if active:
        add_opamp_buffer(d, previous)
    else:
        output(d, previous)
    return f'buffered_RC_low_pass_{n}_stage' if active else f'RC_low_pass_{n}_stage'


def make_high_pass(d, p, level, active=False):
    source(d)
    n = stage_count(level)
    previous = (0, 4)
    for i in range(n):
        x = 4.5 + 4.5 * i
        start = (x - 2.5, 4)
        node = (x, 4)
        line(d, previous, start)
        p.add(d, 'C', start, node)
        p.add(d, 'R', node, (x, 0))
        dot(d, node)
        previous = node
    line(d, (0, 0), (previous[0], 0))
    if active:
        add_opamp_buffer(d, previous)
    else:
        output(d, previous)
    return f'buffered_RC_high_pass_{n}_stage' if active else f'RC_high_pass_{n}_stage'


def make_band_pass(d, p, level, active=False):
    """Series LC feeding a resistive load, optionally repeated as a ladder."""
    source(d)
    n = stage_count(level)
    previous = (0, 4)
    for i in range(n):
        x = 7.0 + 6.5 * i
        a = (x - 5.0, 4)
        b = (x - 2.5, 4)
        c = (x, 4)
        line(d, previous, a)
        if random.choice([True, False]):
            p.add(d, 'L', a, b)
            p.add(d, 'C', b, c)
        else:
            p.add(d, 'C', a, b)
            p.add(d, 'L', b, c)
        p.add(d, 'R', c, (x, 0))
        dot(d, c)
        previous = c
    line(d, (0, 0), (previous[0], 0))
    if active:
        add_opamp_buffer(d, previous)
    else:
        output(d, previous)
    return f'buffered_LC_band_pass_{n}_stage' if active else f'LC_band_pass_{n}_stage'


BUILDERS = {
    'voltage_divider': make_voltage_divider,
    'current_divider': make_current_divider,
    'low_pass_filter': make_low_pass,
    'high_pass_filter': make_high_pass,
    'band_pass_filter': make_band_pass,
}


def next_number(folder):
    ids = []
    for path in folder.glob('circuit_*.png'):
        match = re.match(r'circuit_(\d+)_', path.name)
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=0) + 1


def render_upright_labels(d, path):
    """Rotate/mirror schematic geometry, then position upright text safely.

    Text placement uses the final rendered geometry as an obstacle map.
    The final image is padded so no labels can be clipped at its edges.
    """
    from PIL import ImageStat

    figure_wrapper = d.draw(show=False)
    fig, ax = figure_wrapper.fig, figure_wrapper.ax
    fig.canvas.draw()
    fig.savefig(str(path), dpi=fig.dpi, facecolor='#f2f2f2')
    with Image.open(path) as im:
        base = im.convert('RGB')
    w, h = base.size
    mapped = []
    for (x, y), msg in LABELS:
        px, py = ax.transData.transform((x, y))
        mapped.append((float(px), float(h - py), msg))

    angle = random.choice([0, 90, 180, 270]) if RANDOM_ORIENTATION else 0
    mirrored = random.choice([True, False]) if RANDOM_ORIENTATION else False
    if mirrored:
        base = ImageOps.mirror(base)
        mapped = [(w - x, y, msg) for x, y, msg in mapped]
    if angle == 90:
        mapped = [(y, w - x, msg) for x, y, msg in mapped]
    elif angle == 180:
        mapped = [(w - x, h - y, msg) for x, y, msg in mapped]
    elif angle == 270:
        mapped = [(h - y, x, msg) for x, y, msg in mapped]
    if angle:
        base = base.rotate(angle, expand=True, fillcolor='#f2f2f2')

    # Padding is added BEFORE placing labels. This is especially important
    # for source labels at the left edge and Vout at the right edge.
    margin = 125
    canvas = Image.new('RGB', (base.width + 2 * margin, base.height + 2 * margin), '#f2f2f2')
    canvas.paste(base, (margin, margin))
    mapped = [(x + margin, y + margin, msg) for x, y, msg in mapped]

    font = font_for(17)
    draw = ImageDraw.Draw(canvas)
    occupied = []
    # Preserve a copy of geometry: labels placed earlier must not erase wires.
    geometry = canvas.convert('L')

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    for cx, cy, msg in mapped:
        bbox = draw.textbbox((0, 0), msg, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Candidate positions relative to the actual transformed symbol.
        # Longer distances provide fallbacks for dense ladders.
        offsets = [
            (0, -37), (0, 37), (-tw / 2 - 35, 0), (tw / 2 + 35, 0),
            (0, -59), (0, 59), (-tw / 2 - 55, -27),
            (tw / 2 + 55, -27), (-tw / 2 - 55, 27),
            (tw / 2 + 55, 27), (0, -83), (0, 83),
            (-tw / 2 - 80, 0), (tw / 2 + 80, 0),
        ]
        # The source and output labels are more readable off to the side.
        if msg.startswith(('Vin', 'Iin', 'Vout')):
            offsets = [(-tw / 2 - 42, 0), (tw / 2 + 42, 0),
                       (0, -43), (0, 43)] + offsets

        best = None
        for rank, (dx, dy) in enumerate(offsets):
            x = int(round(cx + dx - tw / 2))
            y = int(round(cy + dy - th / 2))
            rect = (x - 4, y - 4, x + tw + 4, y + th + 4)
            if rect[0] < 5 or rect[1] < 5 or rect[2] > canvas.width - 5 or rect[3] > canvas.height - 5:
                continue
            # Count dark pixels in the candidate rectangle: these are wires,
            # nodes, symbols, or ground markers that text should avoid.
            region = geometry.crop(rect)
            dark = region.point(lambda v: 1 if v < 155 else 0)
            wire_pixels = sum(dark.get_flattened_data()) if hasattr(dark, "get_flattened_data") else sum(dark.getdata())
            label_collision = sum(overlap(rect, other) for other in occupied)
            score = wire_pixels * 25 + label_collision * 35 + rank * 3
            if best is None or score < best[0]:
                best = (score, x, y, rect)
        if best is None:
            continue
        _, x, y, rect = best
        # Background only covers the label rectangle after verifying that
        # it is a relatively clear region. Do not mask the actual component.
        draw.rectangle(rect, fill='#f2f2f2')
        draw.text((x, y - bbox[1]), msg, fill='black', font=font)
        occupied.append(rect)

    canvas.save(path, 'PNG')
    return angle, mirrored


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / 'labels_v7_7.csv'
    columns = [
        'filename', 'circuit_type', 'variant', 'complexity',
        'rotation_degrees_ccw', 'mirrored', 'uses_opamp', 'resistor_count',
        'capacitor_count', 'inductor_count'
    ]
    # Use a new CSV so previous versions' column layouts are not mixed.
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    start = next_number(OUTPUT_DIR)
    categories = (CATEGORIES * ((NUM_IMAGES + len(CATEGORIES) - 1) // len(CATEGORIES)))[:NUM_IMAGES]
    random.shuffle(categories)

    with csv_path.open('a', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        for index, category in enumerate(categories, start):
            level = random.choice(['simple', 'medium', 'complex']) if COMPLEXITY == 'mixed' else COMPLEXITY
            if level not in ('simple', 'medium', 'complex'):
                raise ValueError('COMPLEXITY must be mixed, simple, medium, or complex')
            parts = Parts()
            LABELS.clear()
            filename = f'circuit_{index:04d}_{category}.png'
            path = OUTPUT_DIR / filename
            try:
                # All wiring and components are placed in one fixed coordinate system.
                with schemdraw.Drawing(show=False) as d:
                    d.config(unit=3, fontsize=10, color='black', lw=2)
                    uses_opamp = category in ('low_pass_filter', 'high_pass_filter', 'band_pass_filter') and random.random() < OPAMP_PROBABILITY
                    if uses_opamp:
                        variant = BUILDERS[category](d, parts, level, active=True)
                    else:
                        variant = BUILDERS[category](d, parts, level)
                    angle, mirrored = render_upright_labels(d, path)
                plt.close('all')
                writer.writerow({
                    'filename': filename,
                    'circuit_type': category,
                    'variant': variant,
                    'complexity': level,
                    'rotation_degrees_ccw': angle,
                    'mirrored': mirrored,
                    'uses_opamp': uses_opamp,
                    'resistor_count': parts.counts['R'],
                    'capacitor_count': parts.counts['C'],
                    'inductor_count': parts.counts['L'],
                })
                file.flush()
                print(f'OK {filename} | {level} | {angle}° | mirror={mirrored}')
            except Exception as exc:
                plt.close('all')
                print(f'FAILED {filename}: {exc}')
                raise
    print(f'\nGenerated {NUM_IMAGES} images in {OUTPUT_DIR}')
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
