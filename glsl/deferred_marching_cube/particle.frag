#version 450

layout (binding = 1) uniform sampler2D samplerFire;

layout (location = 0) in vec4 inColor;
layout (location = 1) in float inAlpha;
layout (location = 2) in float inRotation;

//layout (location = 0) out vec4 outFragColor;
layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;
void main () 
{
	vec4 color;
	float alpha = (inAlpha <= 1.0) ? inAlpha : 2.0 - inAlpha;
	
	// Rotate texture coordinates
	// Rotate UV	
	float rotCenter = 0.5;
	float rotCos = cos(inRotation);
	float rotSin = sin(inRotation);
	vec2 rotUV = vec2(
		rotCos * (gl_PointCoord.x - rotCenter) + rotSin * (gl_PointCoord.y - rotCenter) + rotCenter,
		rotCos * (gl_PointCoord.y - rotCenter) - rotSin * (gl_PointCoord.x - rotCenter) + rotCenter);

	
	// Flame
	color = texture(samplerFire, rotUV);
	outAlbedo.a = 0.0;
	
	outAlbedo.rgb = color.rgb * inColor.rgb * alpha;	
}