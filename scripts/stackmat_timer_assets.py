"""Reference-shaped timer meshes, in the solve station's existing 7.8-unit footprint.

Artwork is native vector geometry: no Downloads paths or external textures are
needed by the saved library. Local artwork reads from -Y; finished timers face
+Y (the solver). The source photographs are listed in each mesh's metadata.
"""

from __future__ import annotations

import math
import random

import bpy
from mathutils import Matrix


VARIANTS = ("gen2", "gen3", "gen4", "gen5", "gan_timer2")
REFERENCES = dict(zip(VARIANTS, ("gen2.jpg", "gen3.webp", "gen4.jpeg", "gen5.webp", "images.jpeg")))


def rounded(points, iterations=3):
    """Corner cutting preserves the characteristic concave timer silhouettes."""
    for _ in range(iterations):
        points = [
            p
            for a, b in zip(points, points[1:] + points[:1])
            for p in (
                (0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]),
                (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]),
            )
        ]
    return points


def rectangle(x, y, w, h, radius=0.08):
    r = min(radius, w / 2, h / 2)
    return [
        (x + sx * (w / 2 - r) + r * math.cos(a), y + sy * (h / 2 - r) + r * math.sin(a))
        for sx, sy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270))
        for a in (math.radians(start + i * 90 / 8) for i in range(9))
    ]


def fillet(points, radius=0.10):
    """Round corners locally without turning a long trapezoid into an oval."""
    result = []
    for i, (x, y) in enumerate(points):
        prev, nxt = points[i - 1], points[(i + 1) % len(points)]
        a = math.hypot(prev[0] - x, prev[1] - y)
        b = math.hypot(nxt[0] - x, nxt[1] - y)
        ra, rb = min(radius, a * 0.4) / a, min(radius, b * 0.4) / b
        p = (x + (prev[0] - x) * ra, y + (prev[1] - y) * ra)
        q = (x + (nxt[0] - x) * rb, y + (nxt[1] - y) * rb)
        for j in range(7):
            t = j / 6
            result.append(
                (
                    (1 - t) ** 2 * p[0] + 2 * t * (1 - t) * x + t * t * q[0],
                    (1 - t) ** 2 * p[1] + 2 * t * (1 - t) * y + t * t * q[1],
                )
            )
    return result


def ellipse(x, y, rx, ry, n=48):
    return [
        (x + rx * math.cos(i * 2 * math.pi / n), y + ry * math.sin(i * 2 * math.pi / n))
        for i in range(n)
    ]


def inset(points, scale):
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in points]


def clip(subject, boundary):
    """Clip small printed diamonds/dots to a convex sensor panel."""
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        result = []
        if not subject:
            break
        prev = subject[-1]

        def distance(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        dp = distance(prev)
        for point in subject:
            d = distance(point)
            if (d >= 0) != (dp >= 0):
                t = dp / (dp - d)
                result.append(
                    (prev[0] + t * (point[0] - prev[0]), prev[1] + t * (point[1] - prev[1]))
                )
            if d >= 0:
                result.append(point)
            prev, dp = point, d
        subject = result
    return subject


class TimerBuilder:
    def __init__(self, collection, root, variant, material_factory):
        self.collection, self.root, self.variant = collection, root, variant
        self.objects = []
        self.materials = {}
        colors = {
            "black": (0.008, 0.010, 0.013, 1),
            "rubber": (0.021, 0.024, 0.028, 1),
            "seam": (0.003, 0.004, 0.005, 1),
            "silver": (0.48, 0.57, 0.58, 1),
            "cyan_shell": (0.10, 0.48, 0.59, 1),
            "white_shell": (0.63, 0.68, 0.76, 1),
            "blue": (0.008, 0.19, 0.39, 1),
            "blue_light": (0.035, 0.46, 0.72, 1),
            "blue_dark": (0.006, 0.047, 0.11, 1),
            "blue_mid": (0.015, 0.27, 0.57, 1),
            "yellow": (0.95, 0.77, 0.008, 1),
            "red": (0.62, 0.018, 0.026, 1),
            "green": (0.015, 0.22, 0.065, 1),
            "white": (0.88, 0.90, 0.88, 1),
            "lcd": (0.28, 0.34, 0.23, 1),
            "digit": (0.023, 0.033, 0.024, 1),
            "ghost": (0.24, 0.30, 0.21, 1),
            "led_blue": (0.025, 0.12, 0.8, 1),
        }
        for name, color in colors.items():
            self.materials[name] = material_factory(
                f"SolveStation.Stackmat.{variant}.{name}",
                color,
                roughness=0.36 if name not in ("rubber", "lcd") else 0.6,
                metallic=0.38 if name in ("silver", "cyan_shell") else 0.0,
            )

    def polygon(self, name, points, z, mat, depth=0, bevel=0):
        if len(points) < 3:
            return
        # Normalize winding, including mirrored left-hand panels.
        area = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1]))
        if area < 0:
            points = list(reversed(points))
        n = len(points)
        vertices = [(x, y, z) for x, y in points]
        faces = [tuple(range(n))]
        if depth:
            vertices += [(x, y, z - depth) for x, y in points]
            faces.append(tuple(reversed(range(n, 2 * n))))
            faces += [(i, i + n, (i + 1) % n + n, (i + 1) % n) for i in range(n)]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"SolveStation.Timer.{self.variant}.{name}", mesh)
        self.collection.objects.link(obj)
        obj.data.materials.append(self.materials[mat])
        if bevel:
            mod = obj.modifiers.new("Molded edge", "BEVEL")
            mod.width, mod.segments = bevel, 3
        self.objects.append(obj)
        return obj

    def line(self, name, points, z, mat, width=0.012):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = width / 2
        curve.bevel_resolution = 2
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for p, (x, y) in zip(spline.points, points):
            p.co = (x, y, z, 1)
        obj = bpy.data.objects.new(name, curve)
        self.collection.objects.link(obj)
        curve.materials.append(self.materials[mat])
        self.objects.append(obj)
        return obj

    def text(self, text, x, y, z, size, mat, *, italic=False, width=None):
        curve = bpy.data.curves.new(text, "FONT")
        curve.body, curve.align_x, curve.align_y = text, "CENTER", "CENTER"
        curve.size, curve.shear = size, 0.17 if italic else 0
        curve.resolution_u = 4
        obj = bpy.data.objects.new(text, curve)
        obj.location = (x, y, z)
        self.collection.objects.link(obj)
        curve.materials.append(self.materials[mat])
        self.objects.append(obj)
        if width:
            bpy.context.view_layer.update()
            obj.scale.x *= width / max(obj.dimensions.x, 0.001)
        return obj

    def badge(self, x, y, z, radius, side):
        self.polygon("Hand yellow disc", ellipse(x, y, radius, radius), z, "yellow")

        # Four rounded fingers, a broad curved palm, and opposed thumb.
        def transform(points):
            return [(x + side * px * radius, y + py * radius) for px, py in points]

        palm = rounded(
            [
                (-0.52, -0.12),
                (-0.46, -0.55),
                (-0.16, -0.65),
                (0.05, -0.34),
                (0.36, -0.29),
                (0.65, -0.04),
                (0.60, 0.10),
                (0.36, 0.08),
                (0.18, -0.10),
                (-0.17, 0.02),
            ],
            2,
        )
        self.polygon("Hand palm", transform(palm), z + 0.002, "black")
        for i, (fx, fy, length) in enumerate(
            ((-0.44, 0.30, 0.61), (-0.23, 0.39, 0.77), (-0.02, 0.41, 0.81), (0.19, 0.34, 0.66))
        ):
            self.polygon(
                f"Hand finger {i}",
                transform(rectangle(fx, fy, 0.16, length, 0.075)),
                z + 0.002,
                "black",
            )

    def sensor(self, outline, side, style, z=0.306):
        self.polygon("Sensor surround", outline, z - 0.009, "seam", 0.025, 0.008)
        panel = inset(outline, 0.96)
        if sum(a[0] * c[1] - a[1] * c[0] for a, c in zip(panel, panel[1:] + panel[:1])) < 0:
            panel.reverse()
        self.polygon("Sensor printed face", panel, z, "rubber" if style == "gen2" else "blue")
        cx = (min(x for x, y in outline) + max(x for x, y in outline)) / 2
        cy = (min(y for x, y in outline) + max(y for x, y in outline)) / 2
        if style != "gen2":
            rng = random.Random(f"{style}:{side}")
            lo, hi = min(x for x, y in panel), max(x for x, y in panel)
            bottom, top = min(y for x, y in panel), max(y for x, y in panel)
            step = 0.14 if style == "gen3" else 0.17
            for i in range(math.ceil((hi - lo) / step) + 1):
                for j in range(math.ceil((top - bottom) / step) + 1):
                    px, py = lo + i * step, bottom + j * step
                    if style == "gen3":
                        if abs(px) > 3.35:
                            cell = rectangle(px, py, step + 0.002, step + 0.002, 0.001)
                            self.polygon(
                                "Halftone ground", clip(cell, panel), z + 0.001, "blue_dark"
                            )
                            radius = step * (0.45 - 0.22 * min(1, (abs(px) - 3.35) / 0.7))
                            self.polygon(
                                "Halftone dot",
                                clip(ellipse(px, py, radius, radius, 16), panel),
                                z + 0.002,
                                "blue_light",
                            )
                        else:
                            tile = ellipse(px, py, step * 0.56, step * 0.56, 6)
                            self.polygon(
                                "Honeycomb border", clip(tile, panel), z + 0.001, "blue_mid"
                            )
                            self.polygon(
                                "Honeycomb cell",
                                clip(inset(tile, 0.81), panel),
                                z + 0.002,
                                "blue_light",
                            )
                        continue
                    else:
                        tile = [
                            (px, py + step * 0.49),
                            (px - step * 0.49, py),
                            (px, py - step * 0.49),
                            (px + step * 0.49, py),
                        ]
                    mat = rng.choice(("blue", "blue_light", "blue_mid", "blue_dark"))
                    self.polygon("Sensor mosaic", clip(tile, panel), z + 0.001, mat)
            # Fine blue flowing grid, seen on G4/G5's foil print.
            if style in ("gen4", "gen5"):
                for j in range(7):
                    line = [
                        (
                            lo + (hi - lo) * i / 50,
                            bottom + (top - bottom) * (j + 0.5) / 7 + 0.045 * math.sin(i * 0.12),
                        )
                        for i in range(51)
                    ]
                    # Individual short segments clipped like slender polygons.
                    for a, b in zip(line, line[1:]):
                        self.polygon(
                            "Foil grid",
                            clip([a, b, (b[0], b[1] + 0.006), (a[0], a[1] + 0.006)], panel),
                            z + 0.002,
                            "blue_light",
                        )
        self.badge(cx, cy, z + 0.004, 0.35 if style != "gen5" else 0.41, side)

    def button(self, x, y, z, kind, radius=0.14):
        self.polygon(
            "Button silver rim", ellipse(x, y, radius + 0.018, radius + 0.018), z, "silver", 0.018
        )
        self.polygon("Button", ellipse(x, y, radius, radius), z + 0.009, "black", 0.014, 0.005)
        if kind in ("power", "reset"):
            color = "red" if kind == "power" else "yellow"
            start, end = (55, 305) if kind == "power" else (40, 335)
            pts = [
                (
                    x + radius * 0.59 * math.sin(math.radians(a)),
                    y + radius * 0.59 * math.cos(math.radians(a)),
                )
                for a in range(start, end, 8)
            ]
            self.line(kind, pts, z + 0.014, color, 0.024)
            if kind == "power":
                self.line(
                    "Power stem",
                    [(x, y + radius * 0.12), (x, y + radius * 0.80)],
                    z + 0.014,
                    color,
                    0.024,
                )
            else:
                self.polygon(
                    "Reset arrow",
                    [
                        (x + 0.018, y + radius * 0.58),
                        (x + 0.10, y + radius * 0.60),
                        (x + 0.08, y + 0.018),
                    ],
                    z + 0.015,
                    color,
                )
        else:
            self.text(kind, x, y, z + 0.015, radius * 0.52, "white")

    def lcd(self, x, y, w, h, z, digits="0.000"):
        self.polygon(
            "LCD bezel", rectangle(x, y, w + 0.10, h + 0.10, 0.07), z, "black", 0.035, 0.01
        )
        self.polygon("LCD glass", rectangle(x, y, w, h, 0.045), z + 0.005, "lcd")
        height = h * 0.65
        advance = height * 0.67
        start = x - advance * (len(digits.replace(".", "")) - 1) / 2
        dx = start
        for char in digits:
            if char == ".":
                self.polygon(
                    "LCD decimal",
                    rectangle(dx - advance * 0.48, y - height * 0.40, 0.025, 0.025, 0.003),
                    z + 0.007,
                    "digit",
                )
                continue
            sw, sh, t = height * 0.42, height, 0.035 * height / 0.4
            for idx, (px, py, pw, ph) in enumerate(
                (
                    (0, sh / 2, sw, t),
                    (sw / 2, sh / 4, t, sh / 2),
                    (sw / 2, -sh / 4, t, sh / 2),
                    (0, -sh / 2, sw, t),
                    (-sw / 2, -sh / 4, t, sh / 2),
                    (-sw / 2, sh / 4, t, sh / 2),
                    (0, 0, sw, t),
                )
            ):
                self.polygon(
                    "LCD segment",
                    rectangle(dx + px, y + py, pw, ph, min(t / 2, 0.008)),
                    z + 0.007,
                    "ghost" if idx == 6 else "digit",
                )
            dx += advance

    def logo(self, x, y, z, scale=1):
        self.polygon("Speed Stacks logo rim", ellipse(x, y, 0.42 * scale, 0.21 * scale), z, "white")
        self.polygon(
            "Speed Stacks logo blue",
            ellipse(x, y, 0.39 * scale, 0.18 * scale),
            z + 0.001,
            "blue_dark",
        )
        self.text(
            "SPEED",
            x,
            y + 0.055 * scale,
            z + 0.003,
            0.12 * scale,
            "yellow",
            italic=True,
            width=0.55 * scale,
        )
        self.text(
            "STACKS",
            x,
            y - 0.06 * scale,
            z + 0.003,
            0.13 * scale,
            "white",
            italic=True,
            width=0.62 * scale,
        )
        self.text("III", x, y + 0.15 * scale, z + 0.003, 0.12 * scale, "white", italic=True)

    def finish(self):
        # Bake all modifiers/fonts/curves to one mesh per variant. Thousands of
        # print marks therefore don't add thousands of runtime scene objects.
        bpy.ops.object.select_all(action="DESELECT")
        for obj in self.objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.objects[0]
        bpy.ops.object.convert(target="MESH")
        bpy.ops.object.join()
        obj = bpy.context.object
        obj.name = f"SolveStation.Timer.{self.variant}"
        # Keep every design the SAME width as the former 7.8-unit timer.
        xs = [v.co.x for v in obj.data.vertices]
        scale = 7.84 / (max(xs) - min(xs))
        obj.data.transform(Matrix.Diagonal((scale, scale, 1, 1)))
        obj.parent = self.root
        obj.location = (0, 2.15, -1.50)
        obj.rotation_euler.z = math.pi
        obj["timer_variant"] = self.variant
        obj["timer_reference"] = REFERENCES[self.variant]
        obj["timer_asset_version"] = 2
        obj.hide_render = self.variant != "gen5"
        obj.hide_viewport = self.variant != "gen5"
        return obj


def build_timer_variants(collection, root, material_factory):
    for variant in VARIANTS:
        b = TimerBuilder(collection, root, variant, material_factory)
        if variant in ("gen2", "gen3"):
            outline = rounded(
                [
                    (-4, 0.56),
                    (-2.0, 0.56),
                    (-1.67, 0.70),
                    (1.67, 0.70),
                    (2, 0.56),
                    (4, 0.56),
                    (4.18, 0.25),
                    (4.18, -0.30),
                    (3.95, -0.55),
                    (2.0, -0.55),
                    (1.25, -1.03),
                    (-1.25, -1.03),
                    (-2, -0.55),
                    (-3.95, -0.55),
                    (-4.18, -0.30),
                    (-4.18, 0.25),
                ]
            )
            shell = "cyan_shell" if variant == "gen2" else "black"
        elif variant in ("gen4", "gen5"):
            wide = variant == "gen5"
            outline = rounded(
                [
                    (-4, 0.50),
                    (-2, 0.48),
                    (-1.5, 0.86),
                    (0, 0.75),
                    (1.5, 0.86),
                    (2, 0.48),
                    (4, 0.50),
                    (4.2, 0.15),
                    (4.12, -0.43),
                    (3.50, -1.15 if wide else -0.65),
                    (1.3, -0.82),
                    (0.65, -0.58),
                    (0, -0.65),
                    (-0.65, -0.58),
                    (-1.3, -0.82),
                    (-3.5, -1.15 if wide else -0.65),
                    (-4.12, -0.43),
                    (-4.2, 0.15),
                ]
            )
            shell = "black"
        else:
            outline = rectangle(0, 0, 8.3, 1.15, 0.28)
            shell = "white_shell"
        b.polygon("Bottom rubber", inset(outline, 0.985), 0.065, "rubber", 0.06, 0.02)
        b.polygon("Lower housing", outline, 0.155, shell, 0.09, 0.025)
        b.polygon("Housing seam", outline, 0.170, "seam", 0.018, 0.006)
        b.polygon("Upper housing", inset(outline, 0.995), 0.30, shell, 0.128, 0.035)
        # Rubber feet and inset screws remain visible in low camera angles.
        for x in (-3.3, 3.3):
            b.polygon("Rubber foot", ellipse(x, 0, 0.28, 0.22), 0.018, "rubber", 0.018, 0.006)
        if variant == "gan_timer2":
            for side in (-1, 1):
                panel = rectangle(side * 2.85, 0, 2.10, 0.99, 0.20)
                b.polygon("Sensor rim", panel, 0.309, "silver", 0.01)
                b.polygon("Blue touch pad", inset(panel, 0.96), 0.315, "blue_light")
                for i in range(22):
                    px = side * 2.85 - 0.85 + i * 0.08
                    b.line(
                        "Sensor ripple",
                        [(px, -0.34), (px + 0.02, -0.15), (px + 0.02, 0.15), (px, 0.34)],
                        0.317,
                        "blue",
                        0.007,
                    )
                for r in (0.12, 0.17, 0.21):
                    b.line(
                        "Touch ring",
                        ellipse(side * 2.85, 0, r, r, 64) + [(side * 2.85 + r, 0)],
                        0.319,
                        "white",
                        0.009,
                    )
            b.lcd(-0.40, 0.02, 1.18, 0.48, 0.32)
            b.polygon(
                "GAN blue button", ellipse(0.57, -0.05, 0.17, 0.17), 0.33, "blue_light", 0.018, 0.01
            )
            b.text("GAN", -0.4, 0.43, 0.308, 0.17, "silver", width=0.56)
            b.text("TIMER 2", 0.70, 0.32, 0.308, 0.09, "silver")
            port = b.polygon("USB socket", rectangle(0.7, 0, 0.30, 0.12, 0.015), 0, "seam", 0.025)
            port.rotation_euler.x = math.pi / 2
            port.location = (0, -0.575, 0.15)
        else:
            for side in (-1, 1):
                if variant in ("gen2", "gen3"):
                    pad = fillet(
                        [
                            (2.04, 0.49),
                            (3.99, 0.49),
                            (4.10, 0.18),
                            (4.08, -0.29),
                            (3.91, -0.47),
                            (2.04, -0.47),
                        ]
                    )
                else:
                    pad = fillet(
                        [
                            (2.05, 0.43),
                            (3.96, 0.43),
                            (4.10, 0.10),
                            (3.99, -0.39),
                            (3.48, -1.01 if variant == "gen5" else -0.55),
                            (2.11, -0.66),
                        ]
                    )
                b.sensor([(side * x, y) for x, y in pad], side, variant)
            if variant in ("gen2", "gen3"):
                panel = rounded(
                    [
                        (-1.72, 0.57),
                        (1.72, 0.57),
                        (1.78, -0.40),
                        (1.05, -0.86),
                        (-1.05, -0.86),
                        (-1.78, -0.40),
                    ],
                    2,
                )
                b.polygon("Center panel rim", panel, 0.345, "silver", 0.065, 0.018)
                b.polygon(
                    "Center face",
                    inset(panel, 0.965),
                    0.351,
                    "silver" if variant == "gen2" else "blue_light",
                )
                b.lcd(-0.75, 0.06, 1.35, 0.53, 0.36)
                if variant == "gen2":
                    b.button(0.33, 0.14, 0.365, "POWER", 0.19)
                    b.button(1.22, 0.14, 0.365, "RESET", 0.19)
                else:
                    b.button(0.32, 0.10, 0.365, "power", 0.16)
                    b.button(1.23, 0.15, 0.365, "reset", 0.16)
                    b.button(0.72, -0.21, 0.365, "<", 0.09)
                    b.button(1.12, -0.21, 0.365, "SAVE", 0.09)
                for y, mat in ((0.32, "red"), (0.08, "green")):
                    b.polygon(
                        "Status LED rim", ellipse(0.77, y, 0.065, 0.065), 0.36, "black", 0.012
                    )
                    b.polygon(
                        "Status LED", ellipse(0.77, y, 0.045, 0.045), 0.375, mat, 0.012, 0.006
                    )
                b.text(
                    "STACKMAT" if variant == "gen2" else "STACKMAT PRO",
                    0,
                    -0.48,
                    0.357,
                    0.26,
                    "black",
                    italic=True,
                    width=2.6,
                )
                b.text("SPEED STACKS", 0, -0.70, 0.357, 0.13, "black", italic=True, width=1.6)
            else:
                panel = rounded(
                    [
                        (-1.45, 0.73),
                        (0, 0.63),
                        (1.45, 0.73),
                        (1.92, -0.73),
                        (0.60, -0.45),
                        (0, -0.65),
                        (-0.60, -0.45),
                        (-1.92, -0.73),
                    ],
                    2,
                )
                b.polygon("Raised console", panel, 0.39, "black", 0.12, 0.025)
                b.polygon("Silver console", inset(panel, 0.925), 0.399, "silver")
                b.lcd(0, 0.26, 1.35, 0.48, 0.408)
                b.logo(0, -0.35, 0.413, 1.10)
                if variant == "gen4":
                    for y, mat in ((0.30, "red"), (0.04, "green")):
                        b.polygon("LED surround", ellipse(-0.94, y, 0.065, 0.065), 0.405, "black")
                        b.polygon(
                            "LED lens", ellipse(-0.94, y, 0.045, 0.045), 0.415, mat, 0.014, 0.005
                        )
                    b.button(-1.10, -0.28, 0.410, "power", 0.13)
                    b.button(1.10, -0.28, 0.410, "reset", 0.13)
                    b.button(0.94, 0.30, 0.410, ">", 0.08)
                    b.button(0.99, 0.04, 0.410, "SAVE", 0.08)
                else:
                    b.button(1.03, 0.38, 0.410, "power", 0.16)
                    b.button(0.98, -0.12, 0.410, "reset", 0.14)
                    b.polygon(
                        "Blue LED bezel", ellipse(-1.04, 0.43, 0.10, 0.10), 0.405, "black", 0.02
                    )
                    b.polygon(
                        "Blue LED lens",
                        ellipse(-1.04, 0.43, 0.07, 0.07),
                        0.425,
                        "led_blue",
                        0.02,
                        0.008,
                    )
                    b.polygon(
                        "G5 black label field",
                        rounded([(-1.48, 0.14), (-0.88, 0.05), (-0.58, -0.42), (-1.72, -0.60)], 2),
                        0.401,
                        "black",
                    )
                    g = b.text("G", -1.12, -0.14, 0.410, 0.43, "yellow", italic=True)
                    five = b.text("5", -0.85, -0.30, 0.411, 0.42, "red", italic=True)
                    g.data.offset = five.data.offset = 0.012
                    # G5's two separate thumb pads sit on the sloping front lip.
                    for side in (-1, 1):
                        px = side * 2.35
                        before = len(b.objects)
                        pad = rectangle(px, -0.91, 1.65, 0.39, 0.13)
                        b.polygon("Thumb sensor rim", pad, 0.287, "seam", 0.035, 0.01)
                        b.polygon("Thumb blue print", inset(pad, 0.94), 0.293, "blue")
                        for k in range(10):
                            tx = px - 0.65 + k * 0.14
                            b.polygon(
                                "Thumb mosaic",
                                clip(
                                    [
                                        (tx, -0.72),
                                        (tx - 0.18, -0.91),
                                        (tx, -1.10),
                                        (tx + 0.18, -0.91),
                                    ],
                                    inset(pad, 0.94),
                                ),
                                0.295,
                                "blue_light",
                            )
                        b.polygon(
                            "Fingerprint yellow", ellipse(px, -0.91, 0.24, 0.18), 0.297, "yellow"
                        )
                        for r in (0.05, 0.085, 0.12, 0.155, 0.19):
                            pts = [
                                (px + r * math.cos(a), -0.91 + r * 0.76 * math.sin(a))
                                for a in [math.radians(d) for d in range(-50, 260, 10)]
                            ]
                            b.line("Fingerprint ridge", pts, 0.300, "black", 0.012)
                        # Slope the printed plate down toward the viewer.
                        for obj in b.objects[before:]:
                            transform = (
                                Matrix.Translation((0, -0.99, 0.30))
                                @ Matrix.Rotation(math.radians(48), 4, "X")
                                @ Matrix.Translation((0, 0.72, -0.30))
                            )
                            obj.matrix_world = transform @ obj.matrix_world
        b.finish()
