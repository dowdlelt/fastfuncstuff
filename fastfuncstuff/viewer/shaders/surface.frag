#version 440
// Per-fragment volume sampling between white and pial (pycortex's idea): the
// rasteriser interpolates the scanner-space white/pial points across each
// triangle, so the data shows at its own resolution whatever the mesh.
//
// Up to MAX_LAYERS overlays composite bottom to top over the folding shade,
// as the slices composite the stack. Colouring mirrors viewer/colormap.py
// (normalize, quantize, LUT, threshold alpha); surface3d.py:shade_reference
// is its CPU twin. Block layout must match surface3d.py:pack_uniforms.

layout(location = 0) in vec3 vWhite;
layout(location = 1) in vec3 vPial;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in float vFold;
layout(location = 4) in vec4 vColor;
layout(location = 5) in vec2 vAreas;

layout(location = 0) out vec4 fragColor;

struct Layer {
    mat4 texFromMm;
    mat4 statFromMm;
    vec4 cmap;    // lo, hi, threshold, opacity
    vec4 modes;   // sign mode, alpha mode, panes, kind (0 off, 1 values, 2 labels)
    vec4 info;    // LUT row, palette row, outline (0 none, 1 fill + outline, 2 outline only)
};

const int MAX_LAYERS = 4;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
    mat4 viewModel;
    vec4 morph;      // x: 0 = shape A, 1 = shape B
    vec4 depth;      // lo, hi, samples, folding contrast
    vec4 cross;      // crosshair mm, radius
    vec4 crossRgb;
    vec4 extra;      // equivolume on, vertex-map opacity
    Layer layers[MAX_LAYERS];
};

layout(binding = 1) uniform sampler3D value0;
layout(binding = 2) uniform sampler3D value1;
layout(binding = 3) uniform sampler3D value2;
layout(binding = 4) uniform sampler3D value3;
layout(binding = 5) uniform sampler3D stat0;
layout(binding = 6) uniform sampler3D stat1;
layout(binding = 7) uniform sampler3D stat2;
layout(binding = 8) uniform sampler3D stat3;
// One row of 256 per layer.
layout(binding = 9) uniform sampler2D lut;
// Label colours by value, PALETTE_W per row, each label layer's rows stacked:
// FreeSurfer ids run past 14000.
layout(binding = 10) uniform sampler2D palette;
const int PALETTE_W = 4096;

const int MAX_SAMPLES = 16;

bool inside(vec3 t) { return all(greaterThanEqual(t, vec3(0.0))) && all(lessThanEqual(t, vec3(1.0))); }

// Equivolume depth (Waehnert 2014, per-vertex areas as pycortex): area varies
// linearly from white to pial, so the depth enclosing volume fraction a solves
// a quadratic. Twin of surface3d.py:equivolume_fraction.
float equivolume(float a)
{
    float aw = vAreas.x, ap = vAreas.y, delta = ap - aw;
    if (abs(delta) <= 1e-4 * max(aw + ap, 1e-12)) return a;
    return (sqrt(max((1.0 - a) * aw * aw + a * ap * ap, 0.0)) - aw) / delta;
}

float depthAt(float d) { return extra.x > 0.5 ? equivolume(d) : d; }

float unitOf(float v, vec4 cmap, vec4 modes)
{
    float lo = cmap.x, hi = cmap.y;
    int sgn = int(modes.x + 0.5);
    float top = max(abs(hi), abs(lo));
    if (sgn == 1) return top > 0.0 ? clamp(max(v, 0.0) / top, 0.0, 1.0) : 0.0;
    if (sgn == 2) return top > 0.0 ? clamp(max(-v, 0.0) / top, 0.0, 1.0) : 0.0;
    float span = hi - lo;
    return span > 0.0 ? clamp((v - lo) / span, 0.0, 1.0) : 0.0;
}

float alphaOf(float s, vec4 cmap, vec4 modes)
{
    int sgn = int(modes.x + 0.5);
    int am = int(modes.y + 0.5);
    float mag = sgn == 1 ? max(s, 0.0) : (sgn == 2 ? max(-s, 0.0) : abs(s));
    float thr = abs(cmap.z);
    if (thr <= 0.0) return (sgn == 0 || mag > 0.0) ? 1.0 : 0.0;
    if (mag >= thr) return 1.0;
    if (am == 0) return 0.0;
    float r = clamp(mag / thr, 0.0, 1.0);
    return am == 2 ? r * r : r;
}

vec3 lutColour(float u, vec4 modes, float row)
{
    int panes = int(modes.z + 0.5);
    if (panes > 0) u = (clamp(floor(u * float(panes)), 0.0, float(panes - 1)) + 0.5) / float(panes);
    // Same rounding as apply_colormap: nearest of the LUT's entries.
    ivec2 size = textureSize(lut, 0);
    float idx = clamp(floor(u * float(size.x - 1) + 0.5), 0.0, float(size.x - 1));
    return texelFetch(lut, ivec2(int(idx), int(row + 0.5)), 0).rgb;
}

int labelAt(float d, Layer L, sampler3D vt, vec3 off)
{
    vec3 t = (L.texFromMm * vec4(mix(vWhite, vPial, depthAt(d)) + off, 1.0)).xyz;
    return inside(t) ? int(floor(texture(vt, t).r + 0.5)) : 0;
}

// Labels: nearest (the sampler is), never averaged -- half of region 12 and
// half of region 40 is not region 26. A vote of three depths across the
// ribbon instead of one: a single mid-depth sample landing in a stray voxel
// (white matter, unknown) drew speckled borders all over the inside of
// regions once outlines were on.
int labelVote(Layer L, sampler3D vt, vec3 off)
{
    int label = labelAt(0.5, L, vt, off);
    int lo = labelAt(0.3, L, vt, off);
    int hi = labelAt(0.7, L, vt, off);
    return (lo == hi && lo > 0) ? lo : label;
}

// Depth-averaged value (x) and threshold statistic (y), and whether every
// sample was inside the volume (z), at the fragment shifted by ``off`` mm.
vec3 sampleAvg(Layer L, sampler3D vt, sampler3D st, vec3 off)
{
    int ns = clamp(int(depth.z + 0.5), 1, MAX_SAMPLES);
    float v = 0.0, s = 0.0;
    bool ok = true;
    for (int i = 0; i < MAX_SAMPLES; ++i) {
        if (i >= ns) break;
        float d = ns == 1 ? depth.x : mix(depth.x, depth.y, float(i) / float(ns - 1));
        vec4 p = vec4(mix(vWhite, vPial, depthAt(d)) + off, 1.0);
        vec3 t = (L.texFromMm * p).xyz;
        ok = ok && inside(t);
        v += texture(vt, t).r;
        s += texture(st, (L.statFromMm * p).xyz).r;
    }
    return vec3(v / float(ns), s / float(ns), ok ? 1.0 : 0.0);
}

// One layer over ``col``. Borders are found by sampling one pixel's footprint
// to each side (``dx``, ``dy``: that footprint in scanner mm), not by
// fwidth(): fwidth only compares pixels inside a 2x2 shading quad, so a border
// running along a quad edge vanished -- every border, on a test grid aligned
// with the quads, and gaps in every real ROI outline.
vec3 over(vec3 col, float light, Layer L, sampler3D vt, sampler3D st, vec3 dx, vec3 dy)
{
    int kind = int(L.modes.w + 0.5);
    int outline = int(L.info.z + 0.5);
    vec3 lit = vec3(mix(1.0, light, 0.4));
    if (kind == 2) {
        int label = labelVote(L, vt, vec3(0.0));
        bool border = false;
        if (outline > 0) {
            border = labelVote(L, vt, dx) != label || labelVote(L, vt, -dx) != label
                  || labelVote(L, vt, dy) != label || labelVote(L, vt, -dy) != label;
        }
        ivec2 size = textureSize(palette, 0);
        if (label > 0 && label < PALETTE_W * size.y) {
            int row = int(L.info.y + 0.5) + label / PALETTE_W;
            vec3 rgb = texelFetch(palette, ivec2(label % PALETTE_W, row), 0).rgb * lit;
            if (outline == 0) col = mix(col, rgb, L.cmap.w);
            else if (outline == 1) col = mix(col, border ? rgb * 0.55 : rgb, L.cmap.w);
            else if (border) col = mix(col, rgb, L.cmap.w);
        }
        return col;
    }
    vec3 here = sampleAvg(L, vt, st, vec3(0.0));
    float a = here.z > 0.5 ? alphaOf(here.y, L.cmap, L.modes) : 0.0;
    bool passed = a >= 0.999;
    bool border = false;
    if (outline > 0 && passed) {
        // The suprathreshold region's edge: a neighbour that does not pass.
        border = alphaOf(sampleAvg(L, vt, st, dx).y, L.cmap, L.modes) < 0.999
              || alphaOf(sampleAvg(L, vt, st, -dx).y, L.cmap, L.modes) < 0.999
              || alphaOf(sampleAvg(L, vt, st, dy).y, L.cmap, L.modes) < 0.999
              || alphaOf(sampleAvg(L, vt, st, -dy).y, L.cmap, L.modes) < 0.999;
    }
    vec3 rgb = lutColour(unitOf(here.x, L.cmap, L.modes), L.modes, L.info.x) * lit;
    if (outline == 0) col = mix(col, rgb, a * L.cmap.w);
    else if (outline == 1) col = mix(col, border ? rgb * 0.55 : rgb, a * L.cmap.w);
    else if (border) col = mix(col, rgb, L.cmap.w);
    return col;
}

void main()
{
    vec3 n = normalize(vNormal);
    // Headlight: lit by where the eye is, so every orientation reads.
    float light = 0.35 + 0.65 * abs(n.z);
    // Folding, already scaled to [-1, 1] on the CPU (curv, sulc or binary);
    // positive is sulcal, so darker.
    float base = 0.62 - depth.w * vFold;
    vec3 col = vec3(base) * light;
    // A per-vertex map (thickness, parcellation...) sits on the anatomy and
    // under the volume overlays.
    col = mix(col, vColor.rgb * light, vColor.a * extra.y);

    // Written out rather than looped: sampler arrays indexed by a loop
    // variable are not allowed in GLSL ES 3.0, which the GL backend may use.
    // One pixel's footprint on the surface, in scanner mm, for border tests.
    vec3 mid = mix(vWhite, vPial, 0.5);
    vec3 dx = dFdx(mid), dy = dFdy(mid);
    if (layers[0].modes.w > 0.5) col = over(col, light, layers[0], value0, stat0, dx, dy);
    if (layers[1].modes.w > 0.5) col = over(col, light, layers[1], value1, stat1, dx, dy);
    if (layers[2].modes.w > 0.5) col = over(col, light, layers[2], value2, stat2, dx, dy);
    if (layers[3].modes.w > 0.5) col = over(col, light, layers[3], value3, stat3, dx, dy);

    if (cross.w > 0.0) {
        // A dot in the crosshair colour with a dark rim, so it reads over
        // any overlay colour, including its own.
        float dd = distance(mix(vWhite, vPial, 0.5), cross.xyz);
        if (dd < cross.w) col = mix(col, crossRgb.rgb, 0.9);
        else if (dd < 1.4 * cross.w) col = mix(col, vec3(0.0), 0.75);
    }
    fragColor = vec4(col, 1.0);
}
