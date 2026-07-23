use std::env;
use std::path::PathBuf;

fn main() {
	// Link the system ncnn (NCNN_DIR is set in .cargo/config.toml). Headers live
	// in $NCNN_DIR/include/ncnn; libncnn in $NCNN_DIR/lib (or the default linker
	// path). Unlike upstream, we never clone or cmake-build ncnn here.
	println!("cargo:rerun-if-env-changed=NCNN_DIR");
	let ncnn_dir = env::var("NCNN_DIR").unwrap_or_else(|_| "/usr".to_owned());
	println!("cargo:rustc-link-search=native={ncnn_dir}/lib");
	println!("cargo:rustc-link-search=native={ncnn_dir}/lib64");
	println!("cargo:rustc-link-lib=dylib=ncnn");

	let header = PathBuf::from(&ncnn_dir).join("include").join("ncnn").join("c_api.h");
	let bindings = bindgen::Builder::default()
		.header(header.to_str().expect("NCNN_DIR is valid UTF-8"))
		.allowlist_function("ncnn_.*")
		.allowlist_type("ncnn_.*")
		.allowlist_var("NCNN_.*")
		// The ncnn C API only passes opaque pointers, so bindgen's struct
		// layout assertions (which trip on system types like `_IO_FILE`) buy us
		// nothing here.
		.layout_tests(false)
		.generate()
		.expect("generate ncnn bindings");

	let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
	bindings
		.write_to_file(out_dir.join("bindings.rs"))
		.expect("write ncnn bindings");
}
