#version 320 es
precision highp float;
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;
uniform vec4 u_bg_color;
uniform vec2 u_surface_size;
uniform vec2 u_bg_image_size;

void main() {
	if (u_bg_image_size.x <= 0.0 || u_bg_image_size.y <= 0.0) {
		out_color = u_bg_color;
		return;
	}
	// Cover fit: scale so the image covers the surface, crop the overflow centered.
	float scale = max(u_surface_size.x / u_bg_image_size.x, u_surface_size.y / u_bg_image_size.y);
	vec2 shown = u_surface_size / scale;
	vec2 origin = (u_bg_image_size - shown) * 0.5;
	vec2 img_uv = (origin + v_uv * shown) / u_bg_image_size;
	out_color = texture(u_tex, img_uv);
}
