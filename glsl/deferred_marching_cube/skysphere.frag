#version 450

layout (location = 0) in vec2 inUV;

//layout (location = 0) out vec4 outFragColor;
layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;

void main() 
{
	const vec4 gradientStart = vec4(0.24, 0.69, 1, 1.0);
	const vec4 gradientEnd = vec4(0.09, 0.8, 1, 1.0);
	outAlbedo = mix(gradientStart, gradientEnd, min(0.5 - (inUV.t + 0.05), 0.5)/0.15 + 0.5);
}