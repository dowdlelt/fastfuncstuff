#version 440
// Ray marching through the domain box: MIP, MinIP, mean, first hit, composite.
// Modes and their reductions are described in viewer/vox3d.py, whose pick()
// is this loop's CPU twin.

layout(location = 0) in vec3 vWorld;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
    mat4 domainFromMm;
    mat4 ovFromMm;
    mat4 anFromMm;
    vec4 eye;
    vec4 ovShade;
    vec4 anShade;
    vec4 flags;
    vec4 clip;
    vec4 bg;
};

layout(binding = 1) uniform sampler3D ovTex;
layout(binding = 2) uniform sampler3D anTex;
layout(binding = 3) uniform sampler2D lutTex;
layout(binding = 4) uniform sampler3D statTex;

const int MAX_STEPS = 1024;

bool inside(vec3 t)
{
    return all(greaterThanEqual(t, vec3(0.0))) && all(lessThanEqual(t, vec3(1.0)));
}

float score(float v)
{
    int s = int(flags.y + 0.5);
    if (s == 1) return v;
    if (s == 2) return -v;
    return abs(v);
}

vec3 lut(float v)
{
    float u = clamp((v - ovShade.x) / max(ovShade.y - ovShade.x, 1e-20), 0.0, 1.0);
    return texture(lutTex, vec2((u * 255.0 + 0.5) / 256.0, 0.5)).rgb;
}

float anat(vec3 p)
{
    vec3 t = (anFromMm * vec4(p, 1.0)).xyz;
    if (!inside(t)) return 0.0;
    float a = texture(anTex, t).r;
    return clamp((a - anShade.x) / max(anShade.y - anShade.x, 1e-20), 0.0, 1.0);
}

void main()
{
    vec3 e = eye.xyz;
    vec3 a = (domainFromMm * vec4(e, 1.0)).xyz;
    vec3 b = (domainFromMm * vec4(vWorld, 1.0)).xyz;
    vec3 d = b - a;
    // Slab test in the domain's texture space; s runs 0 at the eye to 1 at
    // this back-face fragment, the same along the ray in every space.
    vec3 inv = vec3(
        abs(d.x) > 1e-12 ? 1.0 / d.x : 1e30,
        abs(d.y) > 1e-12 ? 1.0 / d.y : 1e30,
        abs(d.z) > 1e-12 ? 1.0 / d.z : 1e30);
    vec3 t0 = (vec3(0.0) - a) * inv;
    vec3 t1 = (vec3(1.0) - a) * inv;
    vec3 tn = min(t0, t1);
    vec3 tf = max(t0, t1);
    float s0 = max(max(max(tn.x, tn.y), tn.z), 0.0);
    float s1 = min(min(min(tf.x, tf.y), tf.z), 1.0);
    vec3 world = vWorld - e;
    if (dot(clip.xyz, clip.xyz) > 0.0) {
        float f0 = dot(clip.xyz, e) - clip.w;
        float f1 = dot(clip.xyz, world);
        if (abs(f1) < 1e-12) {
            if (f0 > 0.0) s1 = s0;
        } else if (f1 > 0.0) {
            s1 = min(s1, -f0 / f1);
        } else {
            s0 = max(s0, -f0 / f1);
        }
    }
    vec3 back = bg.rgb;
    if (s1 <= s0) {
        fragColor = vec4(back, 1.0);
        return;
    }
    int n = int(clamp(ceil(length(d) * (s1 - s0) * eye.w), 1.0, float(MAX_STEPS)));
    float ds = (s1 - s0) / float(n);
    int mode = int(flags.x + 0.5);
    bool hasOv = flags.z > 0.5;
    bool hasAn = flags.w > 0.5;
    bool hasStat = bg.w > 0.5;
    float thr = ovShade.z;

    float best = -1e30;
    float bestV = 0.0;
    float lowest = 1e30;
    float sum = 0.0;
    float count = 0.0;
    float anSum = 0.0;
    float anCount = 0.0;
    vec3 acc = vec3(0.0);
    float alpha = 0.0;
    bool hit = false;
    vec3 hitColour = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (i >= n) break;
        float s = s0 + (float(i) + 0.5) * ds;
        vec3 p = e + s * world;
        float an = hasAn ? anat(p) : 0.0;
        anSum += an;
        anCount += 1.0;
        float v = 0.0;
        float cut = 0.0;
        bool valid = false;
        if (hasOv) {
            vec3 t = (ovFromMm * vec4(p, 1.0)).xyz;
            if (inside(t)) {
                v = texture(ovTex, t).r;
                cut = hasStat ? texture(statTex, t).r : v;
                valid = true;
            }
        }
        if (mode == 0) {
            float sc = score(cut);
            if (valid && sc > best) { best = sc; bestV = v; }
        } else if (mode == 1) {
            float x = hasOv ? v : an;
            if ((hasOv ? valid : hasAn) && x != 0.0 && x < lowest) lowest = x;
        } else if (mode == 2) {
            float x = hasOv ? v : an;
            if ((hasOv ? valid : hasAn) && x != 0.0) { sum += x; count += 1.0; }
        } else if (mode == 3) {
            float sc = score(cut);
            if (valid && sc >= thr && sc > 0.0) {
                vec3 t = (ovFromMm * vec4(p, 1.0)).xyz;
                vec3 h = 1.0 / vec3(textureSize(ovTex, 0));
                vec3 g = vec3(
                    score(texture(ovTex, t + vec3(h.x, 0, 0)).r) - score(texture(ovTex, t - vec3(h.x, 0, 0)).r),
                    score(texture(ovTex, t + vec3(0, h.y, 0)).r) - score(texture(ovTex, t - vec3(0, h.y, 0)).r),
                    score(texture(ovTex, t + vec3(0, 0, h.z)).r) - score(texture(ovTex, t - vec3(0, 0, h.z)).r));
                // Gradient in texture space to mm: the transpose of the frame.
                vec3 nrm = transpose(mat3(ovFromMm)) * g;
                float lit = 1.0;
                if (dot(nrm, nrm) > 0.0) {
                    nrm = normalize(nrm);
                    lit = 0.35 + 0.65 * abs(dot(nrm, normalize(world)));
                }
                // Coloured by the peak just inside, not the value at the
                // crossing: that is the threshold itself, so every object
                // would come out the colour of the cut.
                vec3 dims = vec3(textureSize(ovTex, 0));
                int extra = int(ceil(2.0 * eye.w / max(max(dims.x, dims.y), dims.z)));
                float peak = v;
                float peakScore = sc;
                for (int j = 1; j <= 16; ++j) {
                    if (j > extra) break;
                    vec3 q = (ovFromMm * vec4(e + (s + float(j) * ds) * world, 1.0)).xyz;
                    if (!inside(q)) break;
                    float w2 = texture(ovTex, q).r;
                    if (score(w2) > peakScore) { peakScore = score(w2); peak = w2; }
                }
                hitColour = lut(peak) * lit;
                hit = true;
                break;
            }
        } else {
            // Front to back. Alphas are per step and scaled by the step count,
            // so the look does not change with the sampling density.
            float k = 64.0 / max(float(n), 1.0);
            if (hasAn && an > 0.0) {
                float aa = clamp(anShade.z * an * an * 0.25 * k, 0.0, 1.0);
                acc += (1.0 - alpha) * aa * vec3(an);
                alpha += (1.0 - alpha) * aa;
            }
            float sc = score(cut);
            if (valid && sc >= thr && sc > 0.0) {
                float u = clamp((sc - thr) / max(abs(ovShade.y) - thr, 1e-20), 0.0, 1.0);
                float ao = clamp(ovShade.w * (0.25 + 0.75 * u) * 0.6 * k, 0.0, 1.0);
                acc += (1.0 - alpha) * ao * lut(v);
                alpha += (1.0 - alpha) * ao;
            }
            if (alpha > 0.985) break;
        }
    }

    float anMean = anCount > 0.0 ? anSum / anCount : 0.0;
    // The anatomy behind a projection: its mean along the ray, stretched so a
    // brain reads as a brain rather than as a faint haze -- in the ink that
    // contrasts with the background, or a light theme draws white on white.
    vec3 ink = dot(back, vec3(0.299, 0.587, 0.114)) > 0.5 ? vec3(0.0) : vec3(1.0);
    vec3 backdrop = mix(back, ink, hasAn ? clamp(anMean * 2.2, 0.0, 1.0) * anShade.z : 0.0);
    vec3 colour = backdrop;
    if (mode == 0) {
        if (hasOv && best >= thr && best > 0.0) colour = mix(backdrop, lut(bestV), ovShade.w);
    } else if (mode == 1) {
        if (lowest < 1e30) {
            colour = hasOv ? mix(backdrop, lut(lowest), ovShade.w) : vec3(lowest);
        }
    } else if (mode == 2) {
        if (count > 0.0) {
            float m = sum / count;
            if (hasOv) {
                if (score(m) >= thr) colour = mix(backdrop, lut(m), ovShade.w);
            } else {
                colour = vec3(clamp(m * 2.2, 0.0, 1.0));
            }
        }
    } else if (mode == 3) {
        if (hit) colour = mix(backdrop, hitColour, ovShade.w);
    } else {
        colour = acc + (1.0 - alpha) * back;
    }
    fragColor = vec4(colour, 1.0);
}
