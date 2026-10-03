#version 440
// Per-fragment volume sampling between white and pial (pycortex's idea): the
// rasteriser interpolates the scanner-space white/pial points across each
// triangle, so the data shows at its own resolution whatever the mesh.
// Colouring mirrors viewer/colormap.py (normalize, quantize, LUT, threshold
// alpha); viewer/surface3d.py:shade_reference is its CPU twin.

layout(location = 0) in vec3 vWhite;
layout(location = 1) in vec3 vPial;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in float vCurv;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
    mat4 viewModel;
    mat4 texFromMm;
    mat4 statFromMm;
    vec4 morph;
    vec4 depth;
    vec4 cmap;
    vec4 modes;
    vec4 cross;
    vec4 crossRgb;
    vec4 spare;
};

layout(binding = 1) uniform sampler3D valueTex;
layout(binding = 2) uniform sampler3D statTex;
layout(binding = 3) uniform sampler2D lut;

const int MAX_SAMPLES = 16;

bool inside(vec3 t) { return all(greaterThanEqual(t, vec3(0.0))) && all(lessThanEqual(t, vec3(1.0))); }

float unitOf(float v)
{
    float lo = cmap.x, hi = cmap.y;
    int sgn = int(modes.x + 0.5);
    float top = max(abs(hi), abs(lo));
    if (sgn == 1) return top > 0.0 ? clamp(max(v, 0.0) / top, 0.0, 1.0) : 0.0;
    if (sgn == 2) return top > 0.0 ? clamp(max(-v, 0.0) / top, 0.0, 1.0) : 0.0;
    float span = hi - lo;
    return span > 0.0 ? clamp((v - lo) / span, 0.0, 1.0) : 0.0;
}

float alphaOf(float s)
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

void main()
{
    vec3 n = normalize(vNormal);
    // Headlight: lit by where the eye is, so every orientation reads.
    float light = 0.35 + 0.65 * abs(n.z);
    // FreeSurfer curvature is positive in sulci: darker there.
    float base = 0.62 - depth.w * clamp(vCurv * 2.5, -1.0, 1.0);
    vec3 col = vec3(base) * light;

    if (modes.w > 0.5) {
        int ns = clamp(int(depth.z + 0.5), 1, MAX_SAMPLES);
        float v = 0.0, s = 0.0;
        bool ok = true;
        for (int i = 0; i < MAX_SAMPLES; ++i) {
            if (i >= ns) break;
            float d = ns == 1 ? depth.x : mix(depth.x, depth.y, float(i) / float(ns - 1));
            vec4 p = vec4(mix(vWhite, vPial, d), 1.0);
            vec3 t = (texFromMm * p).xyz;
            vec3 ts = (statFromMm * p).xyz;
            ok = ok && inside(t);
            v += texture(valueTex, t).r;
            s += texture(statTex, ts).r;
        }
        v /= float(ns);
        s /= float(ns);
        if (ok) {
            float u = unitOf(v);
            int panes = int(modes.z + 0.5);
            if (panes > 0) u = (clamp(floor(u * float(panes)), 0.0, float(panes - 1)) + 0.5) / float(panes);
            // Same rounding as apply_colormap: nearest of the LUT's entries.
            float nlut = float(textureSize(lut, 0).x);
            float idx = clamp(floor(u * (nlut - 1.0) + 0.5), 0.0, nlut - 1.0);
            vec3 rgb = texture(lut, vec2((idx + 0.5) / nlut, 0.5)).rgb;
            float a = alphaOf(s) * cmap.w;
            col = mix(col, rgb * mix(1.0, light, 0.4), a);
        }
    }

    if (cross.w > 0.0) {
        float dd = distance(mix(vWhite, vPial, 0.5), cross.xyz);
        if (dd < cross.w) col = mix(col, crossRgb.rgb, 0.9);
    }
    fragColor = vec4(col, 1.0);
}
