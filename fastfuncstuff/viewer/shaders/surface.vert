#version 440
// Surface view: morph between two drawn shapes, carry white/pial scanner
// positions to the fragment shader, which samples the volume between them.
// The uniform block must match viewer/surface3d.py:pack_uniforms.

layout(location = 0) in vec3 posA;
layout(location = 1) in vec3 posB;
layout(location = 2) in vec3 nrmA;
layout(location = 3) in vec3 nrmB;
layout(location = 4) in vec3 white;
layout(location = 5) in vec3 pial;
layout(location = 6) in float curv;
layout(location = 7) in vec4 vcolor;   // per-vertex map (thickness, parcellation...)
layout(location = 8) in vec2 areas;    // white and pial vertex areas, for equivolume

layout(location = 0) out vec3 vWhite;
layout(location = 1) out vec3 vPial;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out float vCurv;
layout(location = 4) out vec4 vColor;
layout(location = 5) out vec2 vAreas;

layout(std140, binding = 0) uniform Block {
    mat4 mvp;
    mat4 viewModel;
    mat4 texFromMm;
    mat4 statFromMm;
    vec4 morph;      // x: 0 = shape A, 1 = shape B
    vec4 depth;      // lo, hi, samples, curvature contrast
    vec4 cmap;       // lo, hi, threshold, opacity
    vec4 modes;      // sign mode, alpha mode, panes, has data
    vec4 cross;      // crosshair mm, radius
    vec4 crossRgb;
    vec4 extra;      // equivolume on, vertex-map opacity
};

void main()
{
    vec3 pos = mix(posA, posB, morph.x);
    vec3 nrm = mix(nrmA, nrmB, morph.x);
    vWhite = white;
    vPial = pial;
    vNormal = mat3(viewModel) * nrm;
    vCurv = curv;
    vColor = vcolor;
    vAreas = areas;
    gl_Position = mvp * vec4(pos, 1.0);
}
